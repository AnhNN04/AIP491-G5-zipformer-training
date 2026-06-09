import { describe, it, expect, vi, beforeEach } from 'vitest'
import { uploadAudio, type DecodingConfigInput } from './api_client'

describe('api_client uploadAudio', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('should upload audio file successfully and return transcription', async () => {
    const mockResponse = {
      audio_id: '7a3556d1',
      duration_seconds: 5.2,
      size_bytes: 500,
      status: 'SUCCESS',
      transcription: {
        text: 'chào mừng',
        confidence: 0.99,
        word_alignments: []
      },
      dialect: {
        inferred: 'NORTHERN',
        probability: 0.96
      }
    }

    // Mock global fetch
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    })
    vi.stubGlobal('fetch', fetchMock)

    const file = new File(['fake wav data'], 'speech.wav', { type: 'audio/wav' })
    const config: DecodingConfigInput = { method: 'greedy_search', causal: false }

    const result = await uploadAudio(file, config)

    expect(result.audio_id).toBe('7a3556d1')
    expect(result.transcription.text).toBe('chào mừng')
    expect(fetchMock).toHaveBeenCalledTimes(1)
    
    // Check url and arguments
    const [calledUrl, calledInit] = fetchMock.mock.calls[0]
    expect(calledUrl).toContain('/api/upload')
    expect(calledInit.method).toBe('POST')
    expect(calledInit.body).toBeInstanceOf(FormData)
    
    const bodyFormData = calledInit.body as FormData
    expect(bodyFormData.get('file')).toBe(file)
    expect(bodyFormData.get('config')).toBe(JSON.stringify(config))
  })

  it('should throw error when upload response is not ok', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 413,
      json: async () => ({ detail: 'File size exceeds the 50MB maximum limit.' })
    })
    vi.stubGlobal('fetch', fetchMock)

    const file = new File(['too big audio'], 'big.wav', { type: 'audio/wav' })

    await expect(uploadAudio(file)).rejects.toThrow('File size exceeds the 50MB maximum limit.')
  })
})
