# Natural draft-width completion experiment

This spike compares completed public HTTP requests. It makes no production-source changes and installs no private worker RPC or capture hook.

The request identities live in fixtures.json. The prepare command reconstructs token IDs from the external source and tokenizer, verifies the complete frozen set, and writes an input artifact outside the repository. Raw prompts, model data, caches and results stay outside tracked source.

Each width uses one server load and the same priming, warm-up and repeated request schedule. Requested client concurrency is not an observed internal batch size. Internal batches remain unknown. Ignoring EOS fixes the output cap and is separate from natural speculative acceptance.

The client timer covers the public POST through parsing the complete non-streaming response. It excludes waiting for the client semaphore. Output counts come from response usage. Missing usage, short output, cancellation, HTTP errors and incomplete request sets cannot become survivor-only performance results.

Per-round wall time per output token is the sum of completed request wall times divided by their genuine output-token sum. Throughput uses the actual first-start to last-end span. Total tokens per GPU includes returned input plus output counts and divides by the eight-GPU deployment. These research figures are not official AgentX acceptance results.

The tables retain every round and report the median and observed range. Cache and MTP snapshots are recorded outside each round. Neither those counters nor the client semaphore prove a fixed internal batch.

CPU preflight uses the real ATOM HTTP schema, completion generator, response builder and server argument parser. See test_http.py. Tests also exercise invalid inputs, incomplete output, insufficient repetitions and process-identity reuse.

The controller prints proposed commands by default. Execution requires an explicit reviewed GPU lease, a clean committed head, frozen model/input/cache receipts, the ownership lock, the canonical pause check and idle VRAM. Cleanup keeps ownership until known processes and allocations have been released. Unexpected failures stop the experiment without automatic retry.

Use the approved private environments and the current-source cache receipts. Native build-only settings must not leak into the server. The current compiler consumes the FlyDSL cache, so a historical Mono index directory is not proof of a hit.

The task record and measured reproduction evidence belong in the experimental PR and issue. CPU preparation does not complete the spike. The spike is reproduced independently at its head and closed rather than merged.
