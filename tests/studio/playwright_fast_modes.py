"""Independent Fast control: mocked provider activation, native tiers, and guards."""
import os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

out = Path(os.environ.get('PW_OUT', 'logs/pr-fast-independent'))
out.mkdir(parents=True, exist_ok=True)
base = f"http://127.0.0.1:{os.environ.get('PW_PORT', '5419')}/smoke-fast-modes.html?fast=1"
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 960, 'height': 1100}, device_scale_factor=2, reduced_motion='reduce')
    errors = []
    updates = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    def update(route):
        updates.append(route.request.post_data_json['models'])
        route.fulfill(json={'id': 'smoke', 'models': updates[-1]})
    page.route('**/api/providers/smoke', update)
    for theme in ('dark', 'light'):
        for tier in (False, True):
            page.goto(f"{base}&theme={theme}&" + ('tier=1' if tier else 'gated=1'))
            page.add_style_tag(content='main > output { visibility: hidden; }')
            trigger = page.get_by_role('button', name='Fast mode off', exact=True)
            trigger.click()
            panel = page.get_by_role('dialog', name='Fast settings')
            toggle = panel.get_by_role('switch', name='Fast mode')
            expect(toggle).to_be_enabled()
            toggle.click()
            expect(toggle).to_be_checked()
            expect(page.get_by_role('button', name='Fast mode on', exact=True)).to_be_visible()
            expect(page.get_by_label('Selected model')).to_contain_text('claude-opus-5' if tier else 'mimo-v2.6-pro-ultraspeed')
            expect(panel.get_by_text('$10' if tier else '$4.35', exact=True)).to_be_visible()
            page.wait_for_timeout(250)
            panel.screenshot(path=str(out / f"{'native' if tier else 'companion'}-{theme}-960.png"))
            page.keyboard.press('Escape')
            expect(panel).not_to_be_visible()
            active = page.get_by_role('button', name='Fast mode on', exact=True)
            expect(active).to_be_focused()
            active.screenshot(path=str(out / f"indicator-{'native' if tier else 'companion'}-{theme}-960.png"))
            active.click()
            toggle.click()
            expect(toggle).not_to_be_checked()
    assert updates == [['xiaomi/mimo-v2.6-pro', 'xiaomi/mimo-v2.6-pro-ultraspeed']] * 2
    for state in ('busy', 'unavailable'):
        page.goto(f'{base}&{state}=1')
        page.get_by_role('button', name='Fast mode off', exact=True).click()
        expect(page.get_by_role('switch', name='Fast mode')).to_be_disabled()
    page.goto(f'{base}&gated=1')
    page.route('**/api/providers/smoke', lambda route: route.fulfill(status=503, json={'detail': 'Temporary failure'}))
    page.get_by_role('button', name='Fast mode off', exact=True).click()
    toggle = page.get_by_role('switch', name='Fast mode')
    toggle.click()
    expect(toggle).to_be_enabled()
    expect(toggle).not_to_be_checked()
    expect(page.get_by_label('Selected model')).not_to_contain_text('ultraspeed')
    page.unroute('**/api/providers/smoke')
    page.route('**/api/providers/smoke', update)
    toggle.click()
    expect(toggle).to_be_checked()
    assert not errors, errors
    browser.close()
print('PASS: independent Fast companion activation, native tier, exact model, prices, busy/unavailable guards, retry, focus and screenshots')
