import { ErrorState, LoadingState } from "./PageState";

/** Renders loading, error and ready states for one independently fetched block, so one failing endpoint never blanks a page. */
export default function Async<T>({
  state,
  what,
  children,
}: {
  state: { data: T | null; error: string | null; loading: boolean };
  what: string;
  children: (data: T) => React.ReactNode;
}) {
  if (state.error) return <ErrorState message={`Could not load ${what}: ${state.error}`} />;
  if (!state.data) return state.loading ? <LoadingState /> : <ErrorState message={`No ${what} available.`} />;
  return <>{children(state.data)}</>;
}
