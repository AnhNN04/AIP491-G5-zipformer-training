import argparse
import io
import math
import os
import re
import sys
from collections import Counter, defaultdict

parser = argparse.ArgumentParser(
    description="""
    Generate kneser-ney language model as arpa format. By default,
    it will read the corpus from standard input, and output to standard output.
    """
)
parser.add_argument(
    "-ngram-order",
    type=int,
    default=4,
    choices=[1, 2, 3, 4, 5, 6, 7],
    help="Order of n-gram",
)
parser.add_argument("-text", type=str, default=None, help="Path to the corpus file")
parser.add_argument(
    "-lm", type=str, default=None, help="Path to output arpa file for language models"
)
parser.add_argument(
    "-verbose", type=int, default=0, choices=[0, 1, 2, 3, 4, 5], help="Verbose level"
)
args = parser.parse_args()

default_encoding = "latin-1"

strip_chars = " \t\r\n"
whitespace = re.compile("[ \t]+")


class CountsForHistory:
    def __init__(self):
        self.word_to_count = defaultdict(int)
        self.word_to_context = defaultdict(set)
        self.word_to_f = dict()  # discounted probability
        self.word_to_bow = dict()  # back-off weight
        self.total_count = 0

    def words(self):
        return self.word_to_count.keys()

    def __str__(self):
        # e.g. returns ' total=12: 3->4, 4->6, -1->2'
        return " total={0}: {1}".format(
            str(self.total_count),
            ", ".join(
                [
                    "{0} -> {1}".format(word, count)
                    for word, count in self.word_to_count.items()
                ]
            ),
        )

    def add_count(self, predicted_word, context_word, count):
        assert count >= 0

        self.total_count += count
        self.word_to_count[predicted_word] += count
        if context_word is not None:
            self.word_to_context[predicted_word].add(context_word)


class NgramCounts:
    def __init__(self, ngram_order, bos_symbol="<s>", eos_symbol="</s>"):
        assert ngram_order >= 1

        self.ngram_order = ngram_order
        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol

        self.counts = []
        for n in range(ngram_order):
            self.counts.append(defaultdict(lambda: CountsForHistory()))

        self.d = []  # list of discounting factor for each order of ngram

    def add_count(self, history, predicted_word, context_word, count):
        self.counts[len(history)][history].add_count(
            predicted_word, context_word, count
        )

    def add_raw_counts_from_line(self, line):
        if line == "":
            words = [self.bos_symbol, self.eos_symbol]
        else:
            words = [self.bos_symbol] + whitespace.split(line) + [self.eos_symbol]

        for i in range(len(words)):
            for n in range(1, self.ngram_order + 1):
                if i + n > len(words):
                    break
                ngram = words[i : i + n]
                predicted_word = ngram[-1]
                history = tuple(ngram[:-1])
                if i == 0 or n == self.ngram_order:
                    context_word = None
                else:
                    context_word = words[i - 1]

                self.add_count(history, predicted_word, context_word, 1)

    def add_raw_counts_from_standard_input(self):
        lines_processed = 0
        # byte stream as input
        infile = io.TextIOWrapper(sys.stdin.buffer, encoding=default_encoding)
        for line in infile:
            line = line.strip(strip_chars)
            self.add_raw_counts_from_line(line)
            lines_processed += 1
        if lines_processed == 0 or args.verbose > 0:
            print(
                "make_phone_lm.py: processed {0} lines of input".format(
                    lines_processed
                ),
                file=sys.stderr,
            )

    def add_raw_counts_from_file(self, filename):
        lines_processed = 0
        with open(filename, encoding=default_encoding) as fp:
            for line in fp:
                line = line.strip(strip_chars)
                if self.ngram_order == 1:
                    self.add_raw_counts_from_line(line.split()[0])
                else:
                    self.add_raw_counts_from_line(line)
                lines_processed += 1
        if lines_processed == 0 or args.verbose > 0:
            print(
                "make_phone_lm.py: processed {0} lines of input".format(
                    lines_processed
                ),
                file=sys.stderr,
            )

    def cal_discounting_constants(self):
        self.d = [0]
        for n in range(1, self.ngram_order):
            this_order_counts = self.counts[n]
            n1 = 0
            n2 = 0
            for hist, counts_for_hist in this_order_counts.items():
                stat = Counter(counts_for_hist.word_to_count.values())
                n1 += stat[1]
                n2 += stat[2]
            assert n1 + 2 * n2 > 0
            self.d.append(max(0.1, n1 * 1.0) / (n1 + 2 * n2))

    def cal_f(self):
        n = self.ngram_order - 1
        this_order_counts = self.counts[n]
        for hist, counts_for_hist in this_order_counts.items():
            for w, c in counts_for_hist.word_to_count.items():
                counts_for_hist.word_to_f[w] = (
                    max((c - self.d[n]), 0) * 1.0 / counts_for_hist.total_count
                )

        # lower order N-grams
        for n in range(0, self.ngram_order - 1):
            this_order_counts = self.counts[n]
            for hist, counts_for_hist in this_order_counts.items():
                n_star_star = 0
                for w in counts_for_hist.word_to_count.keys():
                    n_star_star += len(counts_for_hist.word_to_context[w])

                if n_star_star != 0:
                    for w in counts_for_hist.word_to_count.keys():
                        n_star_z = len(counts_for_hist.word_to_context[w])
                        counts_for_hist.word_to_f[w] = (
                            max((n_star_z - self.d[n]), 0) * 1.0 / n_star_star
                        )
                else:  # patterns begin with <s>, they do not have "modified count", so use raw count instead
                    for w in counts_for_hist.word_to_count.keys():
                        n_star_z = counts_for_hist.word_to_count[w]
                        counts_for_hist.word_to_f[w] = (
                            max((n_star_z - self.d[n]), 0)
                            * 1.0
                            / counts_for_hist.total_count
                        )

    def cal_bow(self):
        # highest order N-grams
        n = self.ngram_order - 1
        this_order_counts = self.counts[n]
        for hist, counts_for_hist in this_order_counts.items():
            for w in counts_for_hist.word_to_count.keys():
                counts_for_hist.word_to_bow[w] = None

        # lower order N-grams
        for n in range(0, self.ngram_order - 1):
            this_order_counts = self.counts[n]
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    if w == self.eos_symbol:
                        counts_for_hist.word_to_bow[w] = None
                    else:
                        a_ = hist + (w,)

                        assert len(a_) < self.ngram_order
                        assert a_ in self.counts[len(a_)].keys()

                        a_counts_for_hist = self.counts[len(a_)][a_]

                        sum_z1_f_a_z = 0
                        for u in a_counts_for_hist.word_to_count.keys():
                            sum_z1_f_a_z += a_counts_for_hist.word_to_f[u]

                        sum_z1_f_z = 0
                        _ = a_[1:]
                        _counts_for_hist = self.counts[len(_)][_]
                        # Should be careful here: what is Z1
                        for u in a_counts_for_hist.word_to_count.keys():
                            sum_z1_f_z += _counts_for_hist.word_to_f[u]

                        if sum_z1_f_z < 1:
                            # assert sum_z1_f_a_z < 1
                            counts_for_hist.word_to_bow[w] = (1.0 - sum_z1_f_a_z) / (
                                1.0 - sum_z1_f_z
                            )
                        else:
                            counts_for_hist.word_to_bow[w] = None

    def print_raw_counts(self, info_string):
        # these are useful for debug.
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    ngram = ngram.strip(strip_chars)

                    res.append(
                        "{0}\t{1}".format(ngram, counts_for_hist.word_to_count[w])
                    )
        res.sort(reverse=True)
        for r in res:
            print(r)

    def print_modified_counts(self, info_string):
        # these are useful for debug.
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    ngram = ngram.strip(strip_chars)

                    modified_count = len(counts_for_hist.word_to_context[w])
                    raw_count = counts_for_hist.word_to_count[w]

                    if modified_count == 0:
                        res.append("{0}\t{1}".format(ngram, raw_count))
                    else:
                        res.append("{0}\t{1}".format(ngram, modified_count))
        res.sort(reverse=True)
        for r in res:
            print(r)

    def print_f(self, info_string):
        # these are useful for debug.
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    ngram = ngram.strip(strip_chars)

                    f = counts_for_hist.word_to_f[w]
                    if f == 0:  # f(<s>) is always 0
                        f = 1e-99

                    res.append("{0}\t{1}".format(ngram, math.log(f, 10)))
        res.sort(reverse=True)
        for r in res:
            print(r)

    def print_f_and_bow(self, info_string):
        # these are useful for debug.
        print(info_string)
        res = []
        for this_order_counts in self.counts:
            for hist, counts_for_hist in this_order_counts.items():
                for w in counts_for_hist.word_to_count.keys():
                    ngram = " ".join(hist) + " " + w
                    ngram = ngram.strip(strip_chars)

                    f = counts_for_hist.word_to_f[w]
                    if f == 0:  # f(<s>) is always 0
                        f = 1e-99

                    bow = counts_for_hist.word_to_bow[w]
                    if bow is None:
                        res.append("{1}\t{0}".format(ngram, math.log(f, 10)))
                    else:
                        res.append(
                            "{1}\t{0}\t{2}".format(
                                ngram, math.log(f, 10), math.log(bow, 10)
                            )
                        )
        res.sort(reverse=True)
        for r in res:
            print(r)

    def print_as_arpa(
        self, fout=io.TextIOWrapper(sys.stdout.buffer, encoding="latin-1")
    ):
        # print as ARPA format.

        print("\\data\\", file=fout)
        for hist_len in range(self.ngram_order):
            # print the number of n-grams.
            print(
                "ngram {0}={1}".format(
                    hist_len + 1,
                    sum(
                        [
                            len(counts_for_hist.word_to_f)
                            for counts_for_hist in self.counts[hist_len].values()
                        ]
                    ),
                ),
                file=fout,
            )

        print("", file=fout)

        for hist_len in range(self.ngram_order):
            print("\\{0}-grams:".format(hist_len + 1), file=fout)

            this_order_counts = self.counts[hist_len]
            for hist, counts_for_hist in this_order_counts.items():
                for word in counts_for_hist.word_to_count.keys():
                    ngram = hist + (word,)
                    prob = counts_for_hist.word_to_f[word]
                    bow = counts_for_hist.word_to_bow[word]

                    if prob == 0:  # f(<s>) is always 0
                        prob = 1e-99

                    line = "{0}\t{1}".format("%.7f" % math.log10(prob), " ".join(ngram))
                    if bow is not None:
                        line += "\t{0}".format("%.7f" % math.log10(bow))
                    print(line, file=fout)
            print("", file=fout)
        print("\\end\\", file=fout)


if __name__ == "__main__":
    ngram_counts = NgramCounts(args.ngram_order)

    if args.text is None:
        ngram_counts.add_raw_counts_from_standard_input()
    else:
        assert os.path.isfile(args.text)
        ngram_counts.add_raw_counts_from_file(args.text)

    ngram_counts.cal_discounting_constants()
    ngram_counts.cal_f()
    ngram_counts.cal_bow()

    if args.lm is None:
        ngram_counts.print_as_arpa()
    else:
        with open(args.lm, "w", encoding=default_encoding) as f:
            ngram_counts.print_as_arpa(fout=f)
