import os
import requests
from pathlib import Path

def get_token():
    token = os.environ.get("NASA_EARTHDATA_TOKEN")
    if not token:
        # Check .env
        env_path = Path(".env")
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("NASA_EARTHDATA_TOKEN="):
                        token = line.strip().split("=", 1)[1].strip()
                        os.environ["NASA_EARTHDATA_TOKEN"] = token
                        break
    if not token:
        raise ValueError("NASA_EARTHDATA_TOKEN not found in environment or .env")
    return token

def test_auth():
    token = get_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Paper3-Research-NLDAS-Audit/1.0"
    }

    print("=== TESTING NASA EARTHDATA AUTHENTICATION ===")
    print("Token detected (length: %d chars)" % len(token))

    # Test 1: OPeNDAP Cloud DAS endpoint
    opendap_das_url = (
        "https://opendap.earthdata.nasa.gov/collections/C2033151148-GES_DISC/granules/"
        "NLDAS_FORA0125_H.2.0%3ANLDAS_FORA0125_H.A20230715.1200.020.nc.das"
    )
    print(f"\nTesting OPeNDAP DAS endpoint...")
    r1 = requests.get(opendap_das_url, headers=headers, timeout=30)
    print(f"Status Code: {r1.status_code}")
    if r1.status_code == 200:
        print("SUCCESS! OPeNDAP DAS successfully retrieved.")
        print("DAS Content Sample:\n", r1.text[:2000])
    else:
        print("Response:", r1.text[:500])

    # Test 2: OPeNDAP Cloud DDS endpoint
    opendap_dds_url = (
        "https://opendap.earthdata.nasa.gov/collections/C2033151148-GES_DISC/granules/"
        "NLDAS_FORA0125_H.2.0%3ANLDAS_FORA0125_H.A20230715.1200.020.nc.dds"
    )
    print(f"\nTesting OPeNDAP DDS endpoint...")
    r2 = requests.get(opendap_dds_url, headers=headers, timeout=30)
    print(f"Status Code: {r2.status_code}")
    if r2.status_code == 200:
        print("SUCCESS! OPeNDAP DDS successfully retrieved.")
        print("DDS Content:\n", r2.text)

    # Test 3: HTTPS direct file download (HEAD or range request)
    direct_url = "https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/NLDAS_FORA0125_H.2.0/2023/196/NLDAS_FORA0125_H.A20230715.1200.020.nc"
    print(f"\nTesting HTTPS direct granule endpoint...")
    r3 = requests.head(direct_url, headers=headers, allow_redirects=True, timeout=30)
    print(f"Status Code: {r3.status_code}")
    print("Content-Length:", r3.headers.get("Content-Length"))
    print("Content-Type:", r3.headers.get("Content-Type"))

if __name__ == "__main__":
    test_auth()
