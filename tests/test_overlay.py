"""Browser checks require Chrome and the browser-test extra; see SETUP.md."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p, p.chromium.launch(channel="chrome", headless=True) as instance:
        yield instance


@pytest.fixture
def overlay(browser):
    with browser.new_context(viewport={"width": 720, "height": 1280}) as context:
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        html = (Path(__file__).parents[1] / "src/floptician/static/overlay.html").read_text(encoding="utf-8")
        page.route("http://floptician.test/", lambda route: route.fulfill(body=html, content_type="text/html"))
        page.route("https://fonts.googleapis.com/**", lambda route: route.abort())
        page.route("https://fonts.gstatic.com/**", lambda route: route.abort())
        connections = []
        page.route_web_socket("ws://floptician.test:9001", lambda ws: connections.append(ws))
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        page.clock.install(time=now)
        page.clock.pause_at(now)
        page.goto("http://floptician.test/")
        page.wait_for_function("socket.readyState === WebSocket.OPEN")
        page.evaluate("""() => {
            window.testMessageCount = 0;
            socket.addEventListener('message', () => window.testMessageCount++);
        }""")
        message_count = 0

        def send(board, configuration="SINGLE_ROW"):
            nonlocal message_count
            message_count += 1
            message = {"board": board, "configuration": configuration}
            connections[0].send(json.dumps(message))
            # Wait for the real WebSocket onmessage handler before advancing animations.
            page.wait_for_function(
                "n => window.testMessageCount === n",
                arg=message_count,
            )
            page.clock.run_for(20)

        yield page, send
        assert not errors


def cards(names="Ah Kd Qs", y=1):
    return [{"card": card, "x": i + 1, "y": y} for i, card in enumerate(names.split())]


@pytest.mark.parametrize("port", [9001, 19001])
def test_served_overlay_connects_to_configured_port(browser, served_overlay, port):
    url = served_overlay(port)
    with browser.new_context() as context:
        page = context.new_page()
        connections = []
        page.route("https://fonts.googleapis.com/**", lambda route: route.abort())
        page.route("https://fonts.gstatic.com/**", lambda route: route.abort())
        page.route_web_socket(f"ws://127.0.0.1:{port}", lambda ws: connections.append(ws))
        page.goto(url)
        page.wait_for_function("socket.readyState === WebSocket.OPEN")
        assert len(connections) == 1


def test_confirmed_correction_removes_old_identity_immediately(overlay):
    page, send = overlay
    send(cards())
    page.clock.run_for(600)
    send(cards("Ah Kd Js"))
    page.wait_for_function("document.querySelector('[data-card=Js]') !== null")
    assert page.locator("[data-card=Qs]").count() == 0
    assert page.locator("#board .card").count() == 3
    page.clock.run_for(600)
    assert page.locator("[data-card=Js]").is_visible()


def test_layout_change_resets_chihuahua_row_span(overlay):
    page, send = overlay
    send(cards(y=2), "CHIHUAHUA")
    page.clock.run_for(600)
    assert page.locator("[data-card=Ah]").evaluate("e => e.style.gridRowEnd") == "4"
    send(cards(), "SINGLE_ROW")
    page.clock.run_for(600)
    assert page.locator("[data-card=Ah]").evaluate("e => e.style.gridRowEnd") == "auto"
    assert page.locator("[data-card=Ah]").evaluate("e => e.style.gridRowStart") == "1"


@pytest.mark.parametrize("clear_reason", ["empty", "inactive"])
def test_delayed_clear_cannot_erase_new_board(overlay, clear_reason):
    page, send = overlay
    send(cards())
    if clear_reason == "empty":
        send([])
    else:
        page.clock.run_for(11000)
    send(cards("2h 3d 4s"))
    page.clock.run_for(700)
    assert page.locator("#board .card").count() == 3
    assert page.locator("#board").is_visible()


def test_inactive_overlay_still_clears_after_timeout(overlay):
    page, send = overlay
    send(cards())
    page.clock.run_for(9500)
    assert page.locator("#board").is_visible()
    page.clock.run_for(2100)
    assert page.locator("#board .card").count() == 0
    assert not page.locator("#board").is_visible()


def test_unchanged_board_has_no_dom_writes_and_heartbeats_keep_it_alive(overlay):
    page, send = overlay
    send(cards())
    page.clock.run_for(600)
    page.evaluate("""() => {
        window.mutations = [];
        new MutationObserver(records => mutations.push(...records))
            .observe(document.body, {subtree: true, attributes: true, childList: true, characterData: true});
    }""")
    for _ in range(3):
        page.clock.run_for(5000)
        send(list(reversed(cards())))
    assert page.evaluate("mutations.length") == 0
    assert page.locator("#board .card").count() == 3
    page.clock.run_for(11000)
    assert page.locator("#board .card").count() == 0
    # The identical board must reappear after an inactivity clear.
    send(cards())
    assert page.locator("#board .card").count() == 3


def test_existing_reveal_and_fade_animations_are_preserved(overlay):
    page, send = overlay
    send(cards())
    card = page.locator("[data-card=Ah]")
    assert card.evaluate("e => getComputedStyle(e).animationName") == "revealCard"
    assert card.evaluate("e => getComputedStyle(e).animationDuration") == "0.5s"
    assert card.evaluate("e => getComputedStyle(e).clipPath") != "none"
    assert card.evaluate("e => getComputedStyle(e).transitionProperty") == "all"
    assert card.evaluate("e => getComputedStyle(e).transitionDuration") == "0.3s"
    assert card.evaluate("e => getComputedStyle(e).transitionTimingFunction") == "ease"
    page.clock.run_for(600)
    send([])
    assert card.evaluate("e => getComputedStyle(e).animationName") == "fadeOut"
    assert card.evaluate("e => getComputedStyle(e).animationDuration") == "0.5s"
    page.clock.run_for(600)
    assert page.locator("#board .card").count() == 0
