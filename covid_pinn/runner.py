from .core import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["fit_check","full"])
    parser.add_argument("--cities", default=None, help="Comma-separated city labels")
    parser.add_argument("--skip-us", action="store_true")
    parser.add_argument("--skip-world", action="store_true")
    args, unknown = parser.parse_known_args()

    filt = set(c.strip() for c in args.cities.split(",")) if args.cities else None
    print(f"Device: {DEVICE} | Threads: {N_THREADS} | Mode: {args.mode}")
    summary = []

    if not args.skip_us:
        for label, county, state, sub in US_CITIES:
            if filt and label not in filt: continue
            try:
                df, Npop = load_us_county_series(county, state, sub)
                t0 = time.time()
                run_single_city(label, df, Npop, args.mode)
                summary.append({"city":label,"type":"US","status":"OK","time_min":(time.time()-t0)/60})
            except Exception as e:
                print(f"\n!! FAILED {label}: {e}")
                summary.append({"city":label,"type":"US","status":f"FAIL: {e}","time_min":0})

    if not args.skip_world:
        for city in WORLD_CITIES:
            if filt and city not in filt: continue
            try:
                df, Npop = load_world_city_series(city)
                t0 = time.time()
                run_single_city(city, df, Npop, args.mode)
                summary.append({"city":city,"type":"World","status":"OK","time_min":(time.time()-t0)/60})
            except Exception as e:
                print(f"\n!! FAILED {city}: {e}")
                summary.append({"city":city,"type":"World","status":f"FAIL: {e}","time_min":0})

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    sdf = pd.DataFrame(summary); print(sdf.to_string(index=False))
    sdf.to_csv(Path(OUT_PATH_STR)/"run_summary.csv", index=False)

if __name__ == "__main__":
    main()
