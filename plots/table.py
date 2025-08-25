from pathlib import Path

def read_results(paths):
    results = {}  # folder -> [total, success]

    for path in paths:
        path = Path(path)
        if not path.exists():
            continue

        for folder in path.iterdir():
            if not folder.is_dir():
                continue
            if "dinowm" not in folder.name:
                continue

            total, success = results.get(folder.name, [0, 0])

            for seed_dir in folder.iterdir():
                if not seed_dir.is_dir():
                    continue
                result_file = seed_dir / "results.txt"
                if not result_file.exists():
                    continue

                with result_file.open("r") as f:
                    first_line = f.readline().strip()

                # Expecting something like "... True" or "... False" in the first line
                res = first_line.split(" ")[-1].strip()
                total += 1
                success += (res == "True")

            results[folder.name] = [total, success]

    return results

if __name__ == "__main__":
    #paths = ["./MPC_results", "./MPC_results_2"]
    paths = ["./CEM_RESULTS_2", "./CEM_RESULTS_3", "./CEM_RESULTS"]# "./CEM_RESULTS_4"]
    results = read_results(paths)

    for k, (tot, suc) in results.items():
        rate = (suc / tot) if tot else 0.0
        print(f"{k}: {rate:.4f}  ({suc}/{tot})")
