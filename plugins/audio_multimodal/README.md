# audio_multimodal (`cyberlab_audio`)

Typed WAV datasets: sample-rate/channel checks, path-escape protection, speaker-aware split,
acoustic summary features (RMS, zero-crossing rate, spectral centroid). Standard library and
numpy only. There is no `transcript` view (no licensed ASR model): asking for it raises
`UNAVAILABLE_FEATURE_VIEW`. Fixture: `write_fixture()`. Tests: `tests/security/test_plugins_at16_19.py`.
