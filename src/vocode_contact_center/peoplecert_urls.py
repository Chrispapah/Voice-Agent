"""Official PeopleCert URLs for CertyPal defaults (override via ContactCenterSettings / .env)."""

from __future__ import annotations

HELP_AND_SUPPORT_OLP = "https://www.peoplecert.org/help-and-support-olp"
HELP_AND_SUPPORT_FAQ = "https://www.peoplecert.org/help-and-support/FAQ"
OLP_GUIDELINES_PDF_MAC = "https://passport.peoplecert.org/docs/WebProctoredExamsCandidateGuidelinesMac.pdf"
OLP_GUIDELINES_PDF_WINDOWS = "https://passport.peoplecert.org/docs/WebProctoredExamsCandidateGuidelinesWindows.pdf"
CERTIFICATE_VERIFICATION = "https://www.peoplecert.org/for-corporations/certificate-verification-service/"
TAKE2 = "https://www.peoplecert.org/Take2"
CORPORATE_MEMBERSHIP = "https://www.peoplecert.org/Organizations/Corporate-Membership"
ITIL4_FOUNDATION = (
    "https://www.peoplecert.org/browse-certifications/it-governance-and-service-management/ITIL-1/itil-4-foundation-2565"
)


def system_prompt_url_reference() -> str:
    """Approved URLs the model may cite verbatim (no other peoplecert paths)."""
    return f"""
Approved PeopleCert URLs you may mention when guiding callers (do not invent others):
- Online Proctored help: {HELP_AND_SUPPORT_OLP}
- Help FAQ: {HELP_AND_SUPPORT_FAQ}
- OLP candidate guidelines (Mac PDF): {OLP_GUIDELINES_PDF_MAC}
- OLP candidate guidelines (Windows PDF): {OLP_GUIDELINES_PDF_WINDOWS}
- Certificate verification: {CERTIFICATE_VERIFICATION}
- Take2: {TAKE2}
- Corporate membership: {CORPORATE_MEMBERSHIP}
- Example certification page (ITIL 4 Foundation): {ITIL4_FOUNDATION}
""".strip()
