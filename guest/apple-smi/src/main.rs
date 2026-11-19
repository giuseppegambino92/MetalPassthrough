// apple-smi: the Apple Silicon equivalent of nvidia-smi, for Linux containers running under
// apple/container with MetalPassthrough. Every number comes from the live host (Metal / IORegistry)
// through the MetalPassthrough shared-memory ring.
// Copyright 2026 MetalPassthrough contributors. Licensed under the Apache License, Version 2.0.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_PROCS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy)]
struct DeviceInfo {
    magic: u32,
    gpu_cores: u32,
    name: [c_char; 64],
    metal_family: [c_char; 32],
    host_os: [c_char; 48],
    unified_memory_bytes: u64,
    recommended_working_set: u64,
    max_buffer_length: u64,
    current_allocated: u64,
    max_threads_per_tg: u32,
    threadgroup_memory: u32,
    simd_width: u32,
    has_unified_memory: u32,
    measured_bandwidth_gbs: f64,
    host_page_size: u32,
    abi_version: u32,
    daemon_version: [c_char; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ProcInfo {
    pid: u32,
    session: u32,
    arena_bytes: u64,
    dispatches: u64,
    gpu_ns: u64,
    comm: [c_char; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Stats {
    commands: u64,
    dispatches: u64,
    command_buffers: u64,
    pipeline_compiles: u64,
    pipeline_cache_hits: u64,
    last_kernel_gpu_ns: u64,
    total_gpu_ns: u64,
    arena_bytes: u64,
    arenas: u32,
    active_sessions_vm: u32,
    active_containers: u32,
    doorbells: u32,
    pipelines_resident: u64,
    vm_arena_bytes: u64,
    vm_dispatches: u64,
    vm_total_gpu_ns: u64,
    vm_last_kernel_gpu_ns: u64,
    nprocs: u32,
    _pad: u32,
    procs: [ProcInfo; MAX_PROCS],
}

extern "C" {
    fn mpt_open(out: *mut *mut c_void, arena_bytes: usize) -> c_int;
    fn mpt_last_error() -> *const c_char;
    fn mpt_device(ctx: *mut c_void) -> *const DeviceInfo;
    fn mpt_get_stats(ctx: *mut c_void, out: *mut Stats) -> c_int;
    fn mpt_ping(ctx: *mut c_void, rt_ns: *mut u64) -> c_int;
    fn mpt_unified_addressing(ctx: *mut c_void) -> c_int;
    fn mpt_session_caps(ctx: *mut c_void) -> u64;
    fn mpt_session_id(ctx: *mut c_void) -> u32;
    fn mpt_close(ctx: *mut c_void);
}

fn cstr(b: &[c_char]) -> String {
    unsafe { CStr::from_ptr(b.as_ptr()) }.to_string_lossy().into_owned()
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}
fn mib(b: u64) -> f64 {
    b as f64 / (1u64 << 20) as f64
}

fn meminfo(key: &str) -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)?.parse().ok()))
        .unwrap_or(0)
}

fn fmt_ns(ns: u64) -> String {
    match ns {
        0 => "-".into(),
        n if n < 1_000 => format!("{} ns", n),
        n if n < 1_000_000 => format!("{:.2} us", n as f64 / 1e3),
        n => format!("{:.2} ms", n as f64 / 1e6),
    }
}

fn now_string() -> String {
    // Minimal UTC formatter (no external crates).
    let s = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
    let (days, rem) = (s.div_euclid(86400), s.rem_euclid(86400));
    let (h, m, sec) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    const MON: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    format!("{} {:2} {:02}:{:02}:{:02} {} UTC", MON[(mo - 1) as usize], d, h, m, sec, y)
}

struct Sample {
    info: DeviceInfo,
    stats: Stats,
    uva: bool,
    isolated: bool,
    session: u32,
    lat_min: u64,
    lat_med: u64,
    lat_p99: u64,
    hp_total: u64,
    hp_free: u64,
}

fn sample(ctx: *mut c_void) -> Result<Sample, String> {
    let mut lat = Vec::with_capacity(400);
    for _ in 0..400 {
        let mut ns = 0u64;
        if unsafe { mpt_ping(ctx, &mut ns) } != 0 {
            return Err("ping failed".into());
        }
        lat.push(ns);
    }
    lat.sort_unstable();
    let mut stats: Stats = unsafe { std::mem::zeroed() };
    if unsafe { mpt_get_stats(ctx, &mut stats) } != 0 {
        return Err("stats failed".into());
    }
    Ok(Sample {
        info: unsafe { *mpt_device(ctx) },
        stats,
        uva: unsafe { mpt_unified_addressing(ctx) } != 0,
        isolated: unsafe { mpt_session_caps(ctx) } & 0x8 != 0, // MPT_CAP_PROCESS_ISOLATION
        session: unsafe { mpt_session_id(ctx) },
        lat_min: lat[0],
        lat_med: lat[lat.len() / 2],
        lat_p99: lat[lat.len() * 99 / 100],
        hp_total: meminfo("HugePages_Total:"),
        hp_free: meminfo("HugePages_Free:"),
    })
}

const W: usize = 89;

fn row(cols: &[(&str, usize)]) -> String {
    let mut s = String::from("|");
    for (text, width) in cols {
        let t: String = text.chars().take(*width - 1).collect();
        s.push_str(&format!(" {:<w$}|", t, w = width - 1));
    }
    s
}

fn bar(used: u64, total: u64, width: usize) -> String {
    let f = if total == 0 { 0.0 } else { used as f64 / total as f64 };
    let n = ((f * width as f64).round() as usize).min(width);
    format!("[{}{}] {:5.1}%", "#".repeat(n), ".".repeat(width - n), f * 100.0)
}

fn render(s: &Sample) -> String {
    let i = &s.info;
    let st = &s.stats;
    let sep = |c: char, cols: &[usize]| -> String {
        let mut out = String::from(if c == '=' { "|" } else { "+" });
        for w in cols {
            out.push_str(&c.to_string().repeat(*w));
            out.push(if c == '=' { '|' } else { '+' });
        }
        out
    };
    let cols = [41usize, 24, 22];
    let mut o = Vec::new();
    o.push(now_string());
    o.push(format!("+{}+", "-".repeat(W)));
    o.push(row(&[(
        &format!(
            "APPLE-SMI {}   MetalPassthrough {} (ABI {})   Metal: {}   CUDA bridge: 12.9",
            env!("CARGO_PKG_VERSION"),
            cstr(&i.daemon_version),
            i.abi_version,
            cstr(&i.metal_family)
        ),
        W,
    )]));
    o.push(row(&[(&format!("Host: {}   Guest: Linux {} in apple/container VM", cstr(&i.host_os), uname()), W)]));
    o.push(sep('-', &cols));
    o.push(row(&[("GPU  Name                 Family   Cores", 41), ("Unified Memory", 24), ("Bandwidth  Max TG", 22)]));
    o.push(row(&[("Arena (VM shared, zero-copy)", 41), ("Hugepage pool", 24), ("Containers  Sessions", 22)]));
    o.push(sep('=', &cols));
    o.push(row(&[
        (&format!("  0  {:<20} {:<8} {:>5}", cstr(&i.name), cstr(&i.metal_family), i.gpu_cores), 41),
        (&format!("{:.1} GiB (ws {:.1} GiB)", gib(i.unified_memory_bytes), gib(i.recommended_working_set)), 24),
        (&format!("{:>5.0} GB/s  {:>5}", i.measured_bandwidth_gbs, i.max_threads_per_tg), 22),
    ]));
    let pool = s.hp_total * 2 << 20;
    o.push(row(&[
        (&format!("  {:>7.0} MiB {}", mib(st.vm_arena_bytes), bar(st.vm_arena_bytes, pool, 12)), 41),
        (&format!("{}x2MiB, {} free", s.hp_total, s.hp_free), 24),
        (&format!("{:>10}  {:>8}", st.active_containers, st.active_sessions_vm), 22),
    ]));
    o.push(sep('-', &cols));
    let lookups = st.pipeline_compiles + st.pipeline_cache_hits;
    let hit = if lookups == 0 { 0.0 } else { 100.0 * st.pipeline_cache_hits as f64 / lookups as f64 };
    o.push(row(&[
        (&format!("Pipelines: {} built, {} cached ({:.0}% hit)", st.pipeline_compiles, st.pipeline_cache_hits, hit), 41),
        (&format!("Last kernel: {}", fmt_ns(st.vm_last_kernel_gpu_ns)), 24),
        (&format!("UVA: {}", if s.uva { "VA == dev addr" } else { "translated" }), 22),
    ]));
    o.push(row(&[
        (&format!("Ring RTT min/med/p99: {:.2}/{:.2}/{:.2} us", s.lat_min as f64 / 1e3, s.lat_med as f64 / 1e3, s.lat_p99 as f64 / 1e3), 41),
        (&format!("GPU busy: {}", fmt_ns(st.vm_total_gpu_ns)), 24),
        (&format!("Dispatches: {}", st.vm_dispatches), 22),
    ]));
    o.push(format!("+{}+", "-".repeat(W)));
    o.push(String::new());
    o.push(format!("+{}+", "-".repeat(W)));
    let iso = if s.isolated {
        "GPU isolation: per process (own GPU address space)"
    } else {
        "GPU isolation: shared by all processes in this VM"
    };
    o.push(row(&[(&format!("Processes:{:>width$}", iso, width = W - 12), W)]));
    o.push(row(&[("  Session     PID  Process name          Arena MiB    Dispatches        GPU time", W)]));
    o.push(format!("|{}|", "=".repeat(W)));
    let n = (st.nprocs as usize).min(MAX_PROCS);
    let mut any = false;
    for p in &st.procs[..n] {
        if p.session == s.session {
            continue; // apple-smi's own probe session
        }
        any = true;
        o.push(row(&[(
            &format!(
                "  {:>7} {:>7}  {:<20} {:>10.0} {:>13} {:>15}",
                p.session,
                p.pid,
                cstr(&p.comm),
                mib(p.arena_bytes),
                p.dispatches,
                fmt_ns(p.gpu_ns)
            ),
            W,
        )]));
    }
    if !any {
        o.push(row(&[("  No running processes found", W)]));
    }
    o.push(format!("+{}+", "-".repeat(W)));
    o.join("\n")
}

fn json(s: &Sample) -> String {
    let i = &s.info;
    let st = &s.stats;
    let procs: Vec<String> = st.procs[..(st.nprocs as usize).min(MAX_PROCS)]
        .iter()
        .filter(|p| p.session != s.session)
        .map(|p| {
            format!(
                "{{\"session\":{},\"pid\":{},\"name\":\"{}\",\"arena_bytes\":{},\"dispatches\":{},\"gpu_ns\":{}}}",
                p.session, p.pid, cstr(&p.comm), p.arena_bytes, p.dispatches, p.gpu_ns
            )
        })
        .collect();
    format!(
        "{{\"name\":\"{}\",\"gpu_cores\":{},\"metal_family\":\"{}\",\"host_os\":\"{}\",\"unified_memory_bytes\":{},\
\"recommended_working_set\":{},\"measured_bandwidth_gbs\":{:.1},\"uva\":{},\"process_isolation\":{},\"arena_bytes_vm\":{},\"hugepages_total\":{},\
\"hugepages_free\":{},\"active_containers\":{},\"sessions_vm\":{},\"pipeline_compiles\":{},\"pipeline_cache_hits\":{},\
\"last_kernel_gpu_ns\":{},\"ring_latency_ns\":{{\"min\":{},\"median\":{},\"p99\":{}}},\"processes\":[{}]}}",
        cstr(&i.name),
        i.gpu_cores,
        cstr(&i.metal_family),
        cstr(&i.host_os),
        i.unified_memory_bytes,
        i.recommended_working_set,
        i.measured_bandwidth_gbs,
        s.uva,
        s.isolated,
        st.vm_arena_bytes,
        s.hp_total,
        s.hp_free,
        st.active_containers,
        st.active_sessions_vm,
        st.pipeline_compiles,
        st.pipeline_cache_hits,
        st.vm_last_kernel_gpu_ns,
        s.lat_min,
        s.lat_med,
        s.lat_p99,
        procs.join(",")
    )
}

fn uname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/osrelease").map(|s| s.trim().to_string()).unwrap_or_default()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: apple-smi [-l SECONDS] [--json]\n  Apple GPU status for MetalPassthrough-enabled Linux containers.");
        return;
    }
    let want_json = args.iter().any(|a| a == "--json" || a == "-j");
    let interval = args
        .iter()
        .position(|a| a == "-l" || a == "--loop")
        .and_then(|k| args.get(k + 1))
        .and_then(|v| v.parse::<f64>().ok());
    let mut ctx: *mut c_void = std::ptr::null_mut();
    // Small private arena: apple-smi only needs the control ring.
    if unsafe { mpt_open(&mut ctx, 4 << 20) } != 0 {
        let e = unsafe { CStr::from_ptr(mpt_last_error()) }.to_string_lossy();
        eprintln!("apple-smi: MetalPassthrough not available: {}", e);
        std::process::exit(9);
    }
    loop {
        let t0 = Instant::now();
        match sample(ctx) {
            Ok(s) => {
                if interval.is_some() && !want_json {
                    print!("\x1b[2J\x1b[H");
                }
                println!("{}", if want_json { json(&s) } else { render(&s) });
            }
            Err(e) => {
                eprintln!("apple-smi: {}", e);
                std::process::exit(1);
            }
        }
        match interval {
            Some(sec) => std::thread::sleep(Duration::from_secs_f64(sec).saturating_sub(t0.elapsed())),
            None => break,
        }
    }
    unsafe { mpt_close(ctx) };
}
