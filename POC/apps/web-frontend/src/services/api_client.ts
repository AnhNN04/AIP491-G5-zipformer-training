export interface WordAlignment {
  word: string;
  start: number;
  end: number;
  conf: number;
}

export interface TranscriptionResponse {
  audio_id: string;
  duration_seconds: number;
  size_bytes: number;
  status: string;
  transcription: {
    text: string;
    confidence: number;
    word_alignments: WordAlignment[];
  };
  dialect: {
    inferred: string;
    probability: number;
  };
}

export interface DecodingConfigInput {
  method: 'greedy_search' | 'modified_beam_search';
  beam_size?: number;
  causal?: boolean;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Uploads an audio/video file and calls the transcription API endpoint.
 * Enforces strict TypeScript schemas.
 */
export async function uploadAudio(
  file: File,
  config?: DecodingConfigInput
): Promise<TranscriptionResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  if (config) {
    formData.append('config', JSON.stringify(config));
  }

  const url = `${API_BASE_URL}/api/upload`;
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      let errorMessage = `Upload failed with status: ${response.status}`;
      try {
        const errorBody = await response.json();
        errorMessage = errorBody.detail || errorBody.message || errorMessage;
      } catch {
        // Fall back to HTTP status text if JSON parsing fails
        try {
          const textBody = await response.text();
          if (textBody) {
            errorMessage = textBody;
          }
        } catch {
          // ignore
        }
      }
      throw new Error(errorMessage);
    }

    const data: TranscriptionResponse = await response.json();
    return data;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error('An unexpected network error occurred.');
  }
}
