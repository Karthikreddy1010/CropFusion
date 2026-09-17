import cdsapi

def purge_all_jobs():
    c = cdsapi.Client()
    total_purged = 0
    while True:
        jobs = c.client.get_jobs()
        data = jobs.json if isinstance(jobs.json, dict) else jobs.json()
        job_list = data.get("jobs", [])
        if not job_list:
            print("Queue is completely empty!")
            break
        
        # Check statuses
        active = [j for j in job_list if j.get("status") in ["accepted", "running", "queued"]]
        print(f"Current page has {len(job_list)} jobs (Active: {len(active)})")
        
        ids = [j.get("jobID") for j in job_list if j.get("jobID")]
        for jid in ids:
            try:
                c.client.delete(jid)
                total_purged += 1
            except Exception as e:
                print(f"Error deleting {jid}: {e}")
                
        # Check next page / remaining
        jobs_after = c.client.get_jobs()
        data_after = jobs_after.json if isinstance(jobs_after.json, dict) else jobs_after.json()
        after_list = data_after.get("jobs", [])
        if len(after_list) == len(job_list):
            print(f"No further active jobs. Total purged: {total_purged}. Remaining historical logs: {len(after_list)}")
            break

if __name__ == "__main__":
    purge_all_jobs()
