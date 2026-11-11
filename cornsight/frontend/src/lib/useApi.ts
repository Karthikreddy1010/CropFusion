import { useEffect, useState } from "react";

export interface AsyncState<T> { data: T | null; error: string | null; loading: boolean }

/** Run an API call whenever deps change; ignores stale responses. */
export function useApi<T>(fn: () => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [st, setSt] = useState<AsyncState<T>>({ data: null, error: null, loading: true });
  useEffect(() => {
    let live = true;
    setSt((p) => ({ data: p.data, error: null, loading: true }));
    fn().then(
      (d) => live && setSt({ data: d, error: null, loading: false }),
      (e: Error) => live && setSt({ data: null, error: e.message, loading: false }),
    );
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return st;
}
