#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_sphere4_tdisp.py

Read Rockable conf files and compute tangential displacement involving a target
particle (default: Sphere4) from the Interactions block.

We use:
- interaction normal n
- interaction relativeVelocity v_rel

Tangential relative velocity is computed by:
    v_t_vec = v_rel - (v_rel · n) n

Then different displacement strategies are available:

1) net  : signed/net tangential displacement
          integrate signed scalar tangential velocity
2) abs  : accumulated tangential path length
          integrate |v_t_vec|
3) osc  : oscillatory contribution = abs - |net|
4) each : plot each contact separately

Important note
--------------
For a general 3D contact, "signed tangential displacement" is not unique unless
you choose a tangential direction basis. Here we define a consistent signed
scalar tangential velocity by projecting v_t_vec onto a reference tangential
direction for each contact:
    e_ref = normalized(v_t_vec at first available frame of that contact)
Then:
    v_t_signed = v_t_vec · e_ref

This is very useful for your quasi-1D torsional problem around z, but for a
fully general contact motion it is only a practical convention.

Usage examples
--------------
1) Total net tangential displacement vs time
   python3 plot_sphere4_tdisp.py --start 100 --end 300 --strategy net

2) Total accumulated tangential path vs time
   python3 plot_sphere4_tdisp.py --start 100 --end 300 --strategy abs

3) Plot each contact separately
   python3 plot_sphere4_tdisp.py --start 100 --end 300 --mode each --strategy abs

4) Output CSV
   python3 plot_sphere4_tdisp.py --start 100 --end 300 --csv tdisp.csv
"""

import argparse
import math
from pathlib import Path
import matplotlib.pyplot as plt


EPS = 1e-30


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot tangential displacement from Rockable conf Interactions block."
    )
    p.add_argument("--start", type=int, required=True, help="Start conf index, e.g. 100 for conf100")
    p.add_argument("--end", type=int, required=True, help="End conf index, e.g. 200 for conf200")
    p.add_argument("--step", type=int, default=1, help="Step between conf indices")
    p.add_argument("--directory", type=str, default=".", help="Directory containing conf files")
    p.add_argument("--prefix", type=str, default="conf", help="Filename prefix, default: conf")
    p.add_argument("--particle", type=str, default="Sphere4", help="Particle name to track, default: Sphere4")
    p.add_argument(
        "--xmode",
        type=str,
        choices=["time", "iconf"],
        default="time",
        help="x-axis mode: time (default) or iconf"
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["sum", "mean", "max", "each"],
        default="sum",
        help="How to reduce multiple contacts involving the particle: sum/mean/max/each"
    )
    p.add_argument(
        "--strategy",
        type=str,
        choices=["net", "abs", "osc"],
        default="net",
        help="Displacement strategy: net / abs / osc"
    )
    p.add_argument(
        "--ref-mode",
        type=str,
        choices=["first_vt", "z_tangent"],
        default="first_vt",
        help=(
            "How to define signed tangential direction for 'net': "
            "first_vt = first nonzero tangential direction of that contact; "
            "z_tangent = tangential direction induced by rotation around global z"
        )
    )
    p.add_argument("--outfile", type=str, default="sphere4_tdisp_vs_time.png", help="Output figure filename")
    p.add_argument("--csv", type=str, default="", help="Optional CSV output filename")
    p.add_argument("--show", action="store_true", help="Show the figure interactively")
    return p.parse_args()


# ----------------------------------------------------------------------
# Basic readers
# ----------------------------------------------------------------------

def read_conf_time(conf_path: Path):
    with conf_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("t "):
                parts = s.split()
                if len(parts) >= 2:
                    return float(parts[1])
    raise ValueError(f"Could not find time 't' in {conf_path}")


def read_conf_iconf(conf_path: Path):
    with conf_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("iconf "):
                parts = s.split()
                if len(parts) >= 2:
                    return int(parts[1])
    raise ValueError(f"Could not find 'iconf' in {conf_path}")


def read_particles(conf_path: Path):
    """
    Returns a list of particles:
      particles[idx] = {
          "name": ...,
          "pos": (x,y,z),
          "vel": (vx,vy,vz),
          "acc": (ax,ay,az),
          "Q":   (qw,qx,qy,qz),
          "vrot":(wx,wy,wz),
          "arot":(alphax,alphay,alphaz),
      }
    """
    in_particles = False
    particles = []

    with conf_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()

            if s.startswith("Particles "):
                in_particles = True
                continue

            if not in_particles:
                continue

            if not s or s.startswith("#"):
                continue

            if s.startswith("Interactions ") or s.startswith("Interfaces "):
                break

            parts = s.split()
            if len(parts) < 23:
                continue

            particles.append({
                "name": parts[0],
                "pos": (float(parts[4]), float(parts[5]), float(parts[6])),
                "vel": (float(parts[7]), float(parts[8]), float(parts[9])),
                "acc": (float(parts[10]), float(parts[11]), float(parts[12])),
                "Q": (float(parts[13]), float(parts[14]), float(parts[15]), float(parts[16])),
                "vrot": (float(parts[17]), float(parts[18]), float(parts[19])),
                "arot": (float(parts[20]), float(parts[21]), float(parts[22])),
            })

    if not particles:
        raise ValueError(f"Could not parse Particles block in {conf_path}")

    return particles


def read_particle_index(conf_path: Path, particle_name: str):
    particles = read_particles(conf_path)
    for idx, p in enumerate(particles):
        if p["name"] == particle_name:
            return idx
    raise ValueError(f"Could not find particle '{particle_name}' in {conf_path}")


def read_interactions_for_particle(conf_path: Path, particle_index: int):
    """
    Read all interactions involving the target particle index.

    Expected columns (non-periodic format):
      0  i
      1  j
      2  type
      3  isub
      4  jsub
      5  nx
      6  ny
      7  nz
      8  dn
      9  pos.x
      10 pos.y
      11 pos.z
      12 vel.x
      13 vel.y
      14 vel.z
      15 fn
      16 ft.x
      17 ft.y
      18 ft.z
      19 mom.x
      20 mom.y
      21 mom.z
      22 damp
    """
    in_interactions = False
    out = []

    with conf_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()

            if s.startswith("Interactions "):
                in_interactions = True
                continue

            if not in_interactions:
                continue

            if not s or s.startswith("#"):
                continue

            if s.startswith("Interfaces "):
                break

            parts = s.split()
            if len(parts) < 23:
                continue

            i_idx = int(parts[0])
            j_idx = int(parts[1])
            if i_idx != particle_index and j_idx != particle_index:
                continue

            out.append({
                "i": i_idx,
                "j": j_idx,
                "type": int(parts[2]),
                "isub": int(parts[3]),
                "jsub": int(parts[4]),
                "n": (float(parts[5]), float(parts[6]), float(parts[7])),
                "dn": float(parts[8]),
                "pos": (float(parts[9]), float(parts[10]), float(parts[11])),
                "vrel": (float(parts[18]), float(parts[19]), float(parts[20])),
                "fn": float(parts[21]),
                "ft": (float(parts[22]), float(parts[23]), float(parts[24])),
                "mom": (float(parts[25]), float(parts[26]), float(parts[27])),
                "damp": float(parts[28]),
            })

    return out


# ----------------------------------------------------------------------
# Vector helpers
# ----------------------------------------------------------------------

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


def norm(v):
    return math.sqrt(dot(v, v))


def sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def scale(s, v):
    return (s*v[0], s*v[1], s*v[2])


def normalize(v):
    nv = norm(v)
    if nv < EPS:
        return None
    return (v[0]/nv, v[1]/nv, v[2]/nv)


def tangential_relative_velocity(vrel, n):
    vn = dot(vrel, n)
    return sub(vrel, scale(vn, n))


# ----------------------------------------------------------------------
# Contact key
# ----------------------------------------------------------------------

def contact_key(inter):
    """
    A stable identifier for the same geometric contact across files.
    We use:
      (min(i,j), max(i,j), type, isub/jsub arranged consistently)

    If target particle is either i or j, keep original (i,j,isub,jsub) order to
    preserve side information.
    """
    return (inter["i"], inter["j"], inter["type"], inter["isub"], inter["jsub"])


# ----------------------------------------------------------------------
# Signed tangential direction choices
# ----------------------------------------------------------------------

def signed_direction_first_vt(vt_vec, old_ref):
    """
    Use the first nonzero vt direction as reference.
    """
    if old_ref is not None:
        return old_ref
    e = normalize(vt_vec)
    return e


def signed_direction_z_tangent(inter, particles, particle_index):
    """
    Use the tangential direction induced by rotation around global z at the contact point.
    For target particle p:
        r = contact_pos - p.pos
        ez = (0,0,1)
        tangent reference ~ ez x r
    Then remove any normal component and normalize.
    """
    ez = (0.0, 0.0, 1.0)
    ppos = particles[particle_index]["pos"]
    r = sub(inter["pos"], ppos)
    e = cross(ez, r)

    # ensure tangent to contact plane
    n = inter["n"]
    e = sub(e, scale(dot(e, n), n))
    e = normalize(e)
    return e


# ----------------------------------------------------------------------
# Main logic
# ----------------------------------------------------------------------

def reduce_values(values, mode):
    if not values:
        return None
    if mode == "sum":
        return sum(values)
    if mode == "mean":
        return sum(values) / len(values)
    if mode == "max":
        return max(values)
    raise ValueError(f"Unsupported mode: {mode}")


def main():
    args = parse_args()

    if args.step <= 0:
        raise ValueError("--step must be > 0")

    directory = Path(args.directory)

    # time series storage
    records = []

    # contact cumulative displacement
    cum_net = {}
    cum_abs = {}
    ref_dir = {}   # reference tangent direction for signed projection

    prev_t = None

    for i_conf in range(args.start, args.end + 1, args.step):
        conf_path = directory / f"{args.prefix}{i_conf}"

        if not conf_path.exists():
            print(f"[warning] File not found, skipped: {conf_path}")
            continue

        try:
            t = read_conf_time(conf_path)
            iconf = read_conf_iconf(conf_path)
            particles = read_particles(conf_path)
            pidx = read_particle_index(conf_path, args.particle)
            inters = read_interactions_for_particle(conf_path, pidx)
        except Exception as e:
            print(f"[warning] Failed to parse {conf_path}: {e}")
            continue

        if not inters:
            print(f"[warning] No interaction involving {args.particle} in {conf_path}")
            continue

        dt = 0.0 if prev_t is None else (t - prev_t)
        if dt < 0:
            raise RuntimeError(f"Non-monotonic time detected at {conf_path}")

        current_contact_values = {}

        for inter in inters:
            key = contact_key(inter)
            n = inter["n"]
            vrel = inter["vrel"]
            vt_vec = tangential_relative_velocity(vrel, n)
            vt_abs = norm(vt_vec)

            # initialize storage
            if key not in cum_net:
                cum_net[key] = 0.0
            if key not in cum_abs:
                cum_abs[key] = 0.0
            if key not in ref_dir:
                ref_dir[key] = None

            # choose signed reference direction
            if args.ref_mode == "first_vt":
                ref_dir[key] = signed_direction_first_vt(vt_vec, ref_dir[key])
            elif args.ref_mode == "z_tangent":
                if ref_dir[key] is None:
                    ref_dir[key] = signed_direction_z_tangent(inter, particles, pidx)
                    # fallback if geometry-based tangent is degenerate
                    if ref_dir[key] is None:
                        ref_dir[key] = signed_direction_first_vt(vt_vec, ref_dir[key])
            else:
                raise ValueError(f"Unknown ref-mode: {args.ref_mode}")

            e_ref = ref_dir[key]
            vt_signed = dot(vt_vec, e_ref) if e_ref is not None else 0.0

            # integrate
            if prev_t is not None:
                cum_net[key] += vt_signed * dt
                cum_abs[key] += vt_abs * dt

            if args.strategy == "net":
                current_contact_values[key] = cum_net[key]
            elif args.strategy == "abs":
                current_contact_values[key] = cum_abs[key]
            elif args.strategy == "osc":
                current_contact_values[key] = cum_abs[key] - abs(cum_net[key])
            else:
                raise ValueError(f"Unknown strategy: {args.strategy}")

        x = t if args.xmode == "time" else iconf

        records.append({
            "conf_id": i_conf,
            "time": t,
            "iconf": iconf,
            "x": x,
            "contact_values": current_contact_values,
        })

        prev_t = t

    if not records:
        raise RuntimeError("No valid data found. Please check your conf range and files.")

    # determine all contacts encountered
    all_keys = []
    seen = set()
    for rec in records:
        for k in rec["contact_values"].keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    # reduce data if requested
    xvals = [rec["x"] for rec in records]
    conf_ids = [rec["conf_id"] for rec in records]

    if args.mode == "each":
        each_vals = []
        for rec in records:
            row = []
            for k in all_keys:
                if k in rec["contact_values"]:
                    row.append(rec["contact_values"][k])
                else:
                    row.append(None)
            each_vals.append(row)
    else:
        yvals = []
        for rec in records:
            vals = list(rec["contact_values"].values())
            y = reduce_values(vals, args.mode)
            yvals.append(y)

    # optional CSV
    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            if args.mode == "each":
                if args.xmode == "time":
                    f.write("conf_id,time,contact_id,contact_key,udispl\n")
                    for rec, vals in zip(records, each_vals):
                        for k_id, (k, v) in enumerate(zip(all_keys, vals)):
                            if v is not None:
                                f.write(
                                    f"{rec['conf_id']},{rec['time']:.16e},{k_id},\"{k}\",{v:.16e}\n"
                                )
                else:
                    f.write("conf_id,iconf,contact_id,contact_key,udispl\n")
                    for rec, vals in zip(records, each_vals):
                        for k_id, (k, v) in enumerate(zip(all_keys, vals)):
                            if v is not None:
                                f.write(
                                    f"{rec['conf_id']},{rec['iconf']},{k_id},\"{k}\",{v:.16e}\n"
                                )
            else:
                colname = f"udispl_{args.strategy}_{args.mode}"
                if args.xmode == "time":
                    f.write(f"conf_id,time,{colname}\n")
                    for rec, y in zip(records, yvals):
                        f.write(f"{rec['conf_id']},{rec['time']:.16e},{y:.16e}\n")
                else:
                    f.write(f"conf_id,iconf,{colname}\n")
                    for rec, y in zip(records, yvals):
                        f.write(f"{rec['conf_id']},{rec['iconf']},{y:.16e}\n")
        print(f"[info] CSV written to: {args.csv}")

    # plot
    plt.figure(figsize=(10, 6))

    if args.mode == "each":
        for k_id, k in enumerate(all_keys):
            xs = []
            ys = []
            for rec, vals in zip(records, each_vals):
                v = vals[k_id]
                if v is not None:
                    xs.append(rec["x"])
                    ys.append(v)
            if xs:
                plt.plot(xs, ys, marker="o", linewidth=1, label=f"contact {k_id}")
        plt.legend()

        if args.strategy == "net":
            ylabel = f"net tangential displacement of each contact involving {args.particle}"
            title_y = "each net tangential displacement"
        elif args.strategy == "abs":
            ylabel = f"accumulated tangential path of each contact involving {args.particle}"
            title_y = "each accumulated tangential path"
        else:
            ylabel = f"oscillatory tangential contribution of each contact involving {args.particle}"
            title_y = "each oscillatory tangential contribution"
    else:
        plt.plot(xvals, yvals, marker="o", linewidth=1)

        if args.strategy == "net":
            ylabel = f"{args.mode} net tangential displacement involving {args.particle}"
            title_y = f"{args.mode} net tangential displacement"
        elif args.strategy == "abs":
            ylabel = f"{args.mode} accumulated tangential path involving {args.particle}"
            title_y = f"{args.mode} accumulated tangential path"
        else:
            ylabel = f"{args.mode} oscillatory tangential contribution involving {args.particle}"
            title_y = f"{args.mode} oscillatory tangential contribution"

    if args.xmode == "time":
        plt.xlabel("time t")
        title_x = "time"
    else:
        plt.xlabel("iconf")
        title_x = "iconf"

    plt.ylabel(ylabel)
    plt.title(f"{args.particle} {title_y} vs {title_x}")
    plt.grid(True)
    plt.ylim(0,0.00075)
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=300)
    print(f"[info] Figure written to: {args.outfile}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
