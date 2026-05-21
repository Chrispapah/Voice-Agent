export function formatMicrophoneError(error: unknown): string {
  const name = error instanceof DOMException || error instanceof Error ? error.name : "";
  if (name === "NotAllowedError" || name === "PermissionDeniedError" || name === "SecurityError") {
    return "Microphone access is blocked. Allow microphone access for this site in your browser settings, then try again.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No microphone was found. Connect or enable a microphone, then try again.";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "Your microphone is already in use by another app or browser tab. Close the other app and try again.";
  }
  if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
    return "Your microphone does not support the requested audio settings.";
  }
  return error instanceof Error ? error.message : "Microphone permission denied.";
}
