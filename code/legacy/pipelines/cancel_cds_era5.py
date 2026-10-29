import cdsapi

def cancel_cds_jobs():
    c = cdsapi.Client()
    jobs = c.client.get_jobs()
    data = jobs.json if isinstance(jobs.json, dict) else jobs.json()
    job_list = data.get("jobs", []) if isinstance(data, dict) else []
    print(f"Total jobs in CDS queue: {len(job_list)}")

    era5_land_jobs = []
    for j in job_list:
        process_id = j.get("processID", "")
        if "era5-land" in process_id.lower() or "reanalysis-era5-land" in process_id.lower():
            era5_land_jobs.append(j)

    print(f"ERA5-Land jobs found: {len(era5_land_jobs)}")

    cancelled = []
    not_cancellable = []

    for j in era5_land_jobs:
        jid = j.get("jobID") or j.get("id") or j.get("requestID")
        status = j.get("status")
        process_id = j.get("processID", "")
        print(f"\nTargeting Job: {jid} (Status: {status})")

        try:
            res = c.client.delete(jid)
            print(f"  -> SUCCESS: Deleted/Cancelled {jid}. Response: {res}")
            cancelled.append((jid, status, "CANCELLED/DELETED"))
        except Exception as e:
            print(f"  -> FAILED to cancel {jid}: {e}")
            not_cancellable.append((jid, status, str(e)))

    print("\n" + "=" * 60)
    print("FINAL CDS ERA5-LAND CANCELLATION SUMMARY:")
    print("=" * 60)
    print(f"Total ERA5-Land Jobs Detected : {len(era5_land_jobs)}")
    print(f"Successfully Cancelled/Deleted: {len(cancelled)}")
    print(f"Could Not Cancel              : {len(not_cancellable)}")
    for item in cancelled:
        print(f"  [CANCELLED] {item[0]} (prior status: {item[1]})")
    for item in not_cancellable:
        print(f"  [NOT CANCELLED] {item[0]} (status: {item[1]}): {item[2]}")

if __name__ == "__main__":
    cancel_cds_jobs()
