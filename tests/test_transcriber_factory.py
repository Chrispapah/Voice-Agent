from ai_sdr_agent.transcriber_factory import prefer_nova3_for_greek_browser_stt


def test_prefer_nova3_upgrades_el_nova2() -> None:
    assert prefer_nova3_for_greek_browser_stt("nova-2", "el") == "nova-3"
    assert prefer_nova3_for_greek_browser_stt("nova-2", "el-GR") == "nova-3"


def test_prefer_nova3_keeps_explicit_nova_models() -> None:
    assert prefer_nova3_for_greek_browser_stt("nova-3", "el") == "nova-3"
    assert prefer_nova3_for_greek_browser_stt("nova-3-general", "el") == "nova-3-general"


def test_prefer_nova3_skips_nova2_phonecall() -> None:
    assert prefer_nova3_for_greek_browser_stt("nova-2-phonecall", "el") == "nova-2-phonecall"


def test_prefer_nova3_non_greek_noop() -> None:
    assert prefer_nova3_for_greek_browser_stt("nova-2", "en-US") == "nova-2"
