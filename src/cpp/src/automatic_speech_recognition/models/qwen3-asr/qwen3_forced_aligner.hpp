// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "encoder.hpp"
#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "whisper/feature_extractor.hpp"

namespace ov::genai {

// Normalizes a language name or language code to the aligner's canonical language name.
std::string normalize_alignment_language(const std::string& language);

// Segments transcript text into the alignment units expected by the reference implementation.
std::vector<std::string> segment_alignment_units(const std::string& text, const std::string& canonical_language);

// Repairs non-monotonic timestamp predictions using the reference alignment algorithm.
std::vector<int64_t> fix_timestamp(const std::vector<int64_t>& data);

class Qwen3ForcedAligner {
public:
    Qwen3ForcedAligner(const std::filesystem::path& models_path,
                       const std::string& device,
                       const ov::AnyMap& properties);

    // Aligns the transcript to the chunk audio and returns word timestamps in the original audio timeline.
    std::vector<ASRDecodedResultChunk> align(const std::vector<float>& chunk_wav,
                                             const std::string& text,
                                             const std::string& language,
                                             float chunk_offset_sec);

private:
    std::string resolve_language(const std::string& language) const;
    std::string build_marker_input(const std::vector<std::string>& units, size_t audio_frames) const;

    WhisperFeatureExtractor m_feature_extractor;
    Tokenizer m_tokenizer;
    std::unique_ptr<Qwen3ASREncoder> m_encoder;
    ov::InferRequest m_decoder;

    int64_t m_timestamp_token_id;
    float m_timestamp_segment_ms;
    std::set<std::string> m_supported_languages;
};

}  // namespace ov::genai
