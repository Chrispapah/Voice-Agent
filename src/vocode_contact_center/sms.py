from __future__ import annotations

from vocode_contact_center.settings import ContactCenterSettings


def build_registration_confirmation_message(
    settings: ContactCenterSettings,
    collected_data: dict[str, str],
) -> str:
    full_name = (collected_data.get("full_name") or "there").strip() or "there"
    return settings.registration_confirmation_sms_template.format(full_name=full_name)


def build_generic_follow_up_message(
    settings: ContactCenterSettings,
    collected_data: dict[str, str],
) -> str:
    full_name = (collected_data.get("full_name") or "there").strip() or "there"
    return settings.generic_follow_up_sms_template.format(full_name=full_name)
