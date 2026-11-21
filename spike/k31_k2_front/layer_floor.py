"""Isolated current-build bindings for a whole-K2 producer-free floor."""

import hashlib
from pathlib import Path
from types import FunctionType

_BUILDERS = {}


def omit_oproj(context, slot):
    """Compile-time omission; no producer math, traffic or publication."""


def omit_attnres(context, block, group, on, stamps):
    """The caller supplies exact captured X and post-Oproj prefix."""


def omit_attnres_far(context, block, on, stamps):
    """Omit the additional token groups' producer paths as well."""


def provenance():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def isolated_builder(*, floor=False):
    """Clone only the build namespace, never its default module or cache."""
    from atom.models.kimi_k3.mono.kernels import layer as native

    namespace = native.build.__globals__.copy()
    bindings = {"_BUILDS": {}}
    if floor:
        tag = provenance()

        def symbol(stem, **params):
            return native.kernel_symbol(f"{stem}_xready_{tag[:12]}", **params)

        bindings.update(
            stage_oproj=omit_oproj,
            stage_attnres=omit_attnres,
            stage_attnres_far=omit_attnres_far,
            kernel_symbol=symbol,
            _SOURCES=f"{native._SOURCES}:xready:{tag}",
        )
    namespace.update(bindings)
    builder = FunctionType(
        native.build.__code__,
        namespace,
        native.build.__name__,
        native.build.__defaults__,
        native.build.__closure__,
    )
    builder.__annotations__ = dict(native.build.__annotations__)
    if builder.__globals__["_BUILDS"] is native._BUILDS:
        raise RuntimeError("private build cache was not isolated")
    return builder


def build(key):
    if "floor" not in _BUILDERS:
        _BUILDERS["floor"] = isolated_builder(floor=True)
    return _BUILDERS["floor"](key)


def prove_isolation(key, out):
    """Compare real compile-only artifacts and fail closed on any mismatch."""
    import dataclasses
    import inspect
    import json

    from capture import helper

    from atom.models.kimi_k3.mono.kernels import layer as native
    from atom.mono.runtime.compile import (
        _code_object,
        compile_kernels,
        launcher_resources,
    )

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    extract_text = helper("front_floor_compile").elf_text
    protected = (
        "stage_oproj",
        "stage_attnres",
        "stage_attnres_far",
        "kernel_symbol",
        "_SOURCES",
        "_BUILDS",
    )
    before = {name: native.build.__globals__[name] for name in protected}
    result = {
        "state": "STARTED",
        "adapter_sha256": provenance(),
        "artifacts": {},
        "native_source_digest": native._SOURCES,
        "key": dataclasses.asdict(key),
        "abi": list(native.ABI.names),
        "build_code_identity": isolated_builder().__code__ is native.build.__code__,
    }

    def unchanged():
        assert all(
            native.build.__globals__[name] is value for name, value in before.items()
        )

    def collect(label, launcher):
        from flydsl.compiler.kernel_function import KernelFunction

        closure = inspect.getclosurevars(launcher.func)
        functions = {"launch": launcher.func}
        functions.update(
            {
                name: value._func
                for name, value in {**closure.globals, **closure.nonlocals}.items()
                if isinstance(value, KernelFunction)
            }
        )
        if len(functions) < 2:
            raise RuntimeError("kernel generated source is unavailable")
        source = json.dumps(
            {name: inspect.getsource(fn) for name, fn in functions.items()},
            sort_keys=True,
        )
        compile_kernels([(launcher, native.ABI)])
        artifacts = list(launcher._mem_cache.values())
        if len(artifacts) != 1:
            raise RuntimeError("one artifact per isolated build required")
        ir = artifacts[0]._ir_text
        binary = _code_object(ir)
        text = extract_text(binary)
        (out / (label + ".source.json")).write_text(source)
        (out / (label + ".mlir")).write_text(ir)
        (out / (label + ".hsaco")).write_bytes(binary)
        (out / (label + ".text")).write_bytes(text)
        record = {
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "ir_sha256": hashlib.sha256(ir.encode()).hexdigest(),
            "text_sha256": hashlib.sha256(text).hexdigest(),
            "resources": launcher_resources(launcher),
        }
        result["artifacts"][label] = record
        unchanged()
        return source, ir, text

    try:
        baseline = native.build(key)
        a = collect("native_before", baseline)
        original = isolated_builder()
        b = collect("isolated_original", original(key))
        if a != b:
            raise RuntimeError("original-binding generated source/IR/text differ")
        floor = isolated_builder(floor=True)
        f = collect("floor", floor(key))
        if f[2] == a[2]:
            raise RuntimeError("producer-free floor reused baseline compiled text")
        native_cache = dict(native._BUILDS)
        broken = isolated_builder(floor=True)

        def fail(*args):
            raise RuntimeError("intentional private producer failure")

        broken.__globals__["stage_oproj"] = fail
        broken.__globals__["_SOURCES"] += ":negative-control"
        try:
            compile_kernels([(broken(key), native.ABI)])
        except Exception as error:
            if "intentional private producer failure" not in str(error):
                raise
            result["private_failure_detected"] = True
        else:
            raise RuntimeError("compiler ignored the private failure binding")
        unchanged()
        assert native._BUILDS.keys() == native_cache.keys()
        assert all(native._BUILDS[k] is value for k, value in native_cache.items())
        c = collect("isolated_after_failure", isolated_builder()(key))
        d = collect("native_after", native.build(key))
        if a != c or a != d:
            raise RuntimeError("baseline source/IR/text changed after floor/failure")
        result["isa"] = inspect_isa(out)
        result.update(
            state="PASSED", default_bindings_unchanged=True, baseline_text_equal=True
        )
        return result
    finally:
        (out / "identity.json").write_text(json.dumps(result, indent=2) + "\n")


def inspect_isa(out):
    """Inspect the fixed gfx950 ABI and surviving consumer work, without a device."""
    import collections
    import re
    import subprocess

    records = {}
    for label in ("native_before", "floor"):
        assembly = subprocess.check_output(
            [
                "/opt/rocm/llvm/bin/llvm-objdump",
                "--disassemble",
                "--mcpu=gfx950",
                str(out / (label + ".hsaco")),
            ],
            text=True,
        )
        (out / (label + ".s")).write_text(assembly)
        instructions = collections.Counter(
            re.findall(r"^\s+([a-z][a-z0-9_]+) ", assembly, re.MULTILINE)
        )
        records[label] = {
            "assembly_sha256": hashlib.sha256(assembly.encode()).hexdigest(),
            "instructions": dict(instructions),
        }
        if label == "floor":
            prefix = assembly.split("s_waitcnt", 1)[0]
            loads = re.findall(
                r"s_load_dwordx(\d+) s\[\d+:\d+\], s\[0:1\], (0x[0-9a-f]+)", prefix
            )
            consumed = {
                byte
                for words, offset in loads
                for byte in range(int(offset, 16), int(offset, 16) + 4 * int(words))
            }
            if consumed != set(range(64, 200)):
                raise RuntimeError("floor kernarg consumer/producer liveness changed")
            for operation in (
                "v_mfma_f32_16x16x32_bf16",
                "v_mfma_scale_f32_16x16x128_f8f6f4",
                "s_barrier",
                "s_sleep",
                "global_atomic_add",
            ):
                if instructions[operation] == 0:
                    raise RuntimeError(
                        "required consumer operation absent: " + operation
                    )
    records["scope"] = (
        "S4/NB2: producer-only bytes 0..63 are not loaded; downstream weights, output, scratch, plain, queue, peers, rank and layer bytes 64..199 remain. "
        "Native build code and consumer bindings are reused; surviving BF16/MXFP4 matrix operations, barriers, polls and queue atomics are checked. "
        "Static liveness is not numerical correctness; real A/A and A/B validation remains mandatory."
    )
    return records
