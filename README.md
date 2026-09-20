KA08/N15_M01_F10_KA08_2.mat deleted — truncated .mat (522KB zero tail) — corpus = 2559." So future-you / collaborators know why KA08 has 79 files and the corpus is 2559, not 2560.

Data corpus & integrity notes
The raw dataset contains 2560 .mat recordings (32 bearings × 4 operating conditions × 20 repetitions).

Three recordings deviate from the nominal 4-second / 256,000-sample scheme and are excluded:

Exclusion	File	Reason
Corrupt	KA08/N15_M01_F10_KA08_2.mat	Truncated .mat — ~522 KB of trailing zero bytes; fails to load (Expecting matrix here)
Short signal	KA08/N15_M07_F10_KA08_11.mat	Only 249,940 samples (~94 ms shy of nominal) — can't be trimmed to the fixed length
Short signal	KA05/N15_M07_F10_KA05_16.mat	Only 250,604 samples — same reason
Working signal corpus: 2557 recordings (480 healthy / 2077 damaged).

Sampling-length note: recording lengths are not exactly constant. The majority sit at ~256,001 samples, but a meaningful tail runs up to ~299,000 (~22% longer than nominal). All kept recordings are trimmed to a fixed length of 255,996 samples in the cleaning step so that every signal conforms to one schema (match_catalog check passes for 100% of the corpus).

