// k6 benchmark for POST /api/v1/verify (RetinaFace-R34 + EdgeFace Base).
//
// What it measures:
// - end-to-end HTTP latency (k6 http_req_duration)
// - server-side pipeline time via `processing_time_ms` in the verify response
// - success rate (HTTP 200 + expected response shape)
//
// Prerequisites:
// 1. Install k6: https://grafana.com/docs/k6/set-up/install-k6/
// 2. Start the API with models loaded (GET /health -> models_loaded: true):
//      docker compose up --build
// 3. Drop two face photos with clearly visible faces here:
//      benchmark/testdata/target.jpg     (live camera frame)
//      benchmark/testdata/reference.jpg  (reference photo)
//    Same identity = `verified: true` expected; different identities still
//    exercise the full pipeline, only `verified` flips to false.
//    Without real faces the API returns 422 and the benchmark is meaningless.
//
// Run from the repo root:
//   k6 run benchmark/verify.js
//   BASE_URL=http://localhost:8000 k6 run benchmark/verify.js
//   BASE_URL=http://192.168.1.50:8000 k6 run benchmark/verify.js
//
// Tune load by editing `stages` below. Defaults assume the stock
// docker-compose limits (2 CPU, 2 workers): 1 VUs baseline, 6 VUs stress.

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// NOTE: k6 open() resolves relative to this script's directory (benchmark/),
// not your shell cwd. Run from the repo root as shown above.
const targetBin = open('./testdata/target.jpg', 'b');
const referenceBin = open('./testdata/reference.jpg', 'b');

export const options = {
  stages: [
    { duration: '30s', target: 1 }, // warm-up + baseline
    { duration: '1m', target: 3 },
    { duration: '30s', target: 6 }, // stress ramp
    { duration: '1m', target: 6 }, // saturation
    { duration: '30s', target: 0 }, // ramp-down
  ],
  thresholds: {
    http_req_failed: ['rate<0.01'],
    http_req_duration: ['p(95)<5000'],
    verify_processing_ms: ['p(95)<4000'],
  },
};

const processingTime = new Trend('verify_processing_ms');
const verifyOk = new Rate('verify_ok');

export function setup() {
  const res = http.get(`${BASE_URL}/health`, { timeout: '10s' });
  if (res.status !== 200) {
    throw new Error(`health check failed: ${BASE_URL}/health returned status ${res.status}`);
  }
  let body;
  try {
    body = res.json();
  } catch (e) {
    throw new Error(`health check returned non-JSON from ${BASE_URL}/health`);
  }
  if (!body.models_loaded) {
    throw new Error(`models not fully loaded: ${JSON.stringify(body.models)}`);
  }
}

export default function () {
  const payload = {
    target_image: http.file(targetBin, 'target.jpg', 'image/jpeg'),
    reference_image: http.file(referenceBin, 'reference.jpg', 'image/jpeg'),
  };
  const res = http.post(`${BASE_URL}/api/v1/verify`, payload, {
    timeout: '60s',
    tags: { endpoint: 'verify' },
  });

  const ok = check(res, {
    'status is 200': (r) => r.status === 200,
    'has verified flag': (r) => r.status === 200 && typeof r.json('verified') === 'boolean',
    'has similarity_score': (r) => r.status === 200 && typeof r.json('similarity_score') === 'number',
    'has processing_time_ms': (r) => r.status === 200 && typeof r.json('processing_time_ms') === 'number',
  });
  verifyOk.add(ok);

  if (res.status === 200) {
    const ms = res.json('processing_time_ms');
    if (typeof ms === 'number') {
      processingTime.add(ms);
    }
  }

  sleep(1);
}
