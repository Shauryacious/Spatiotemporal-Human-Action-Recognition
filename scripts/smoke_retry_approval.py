"""Offline browser check for retry consent. No worker or provider calls are used."""

import asyncio
import json
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def main():
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT / "src/shar/web"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(channel="chrome", headless=True)
            page = await browser.new_page()
            errors, approvals = [], []
            page.on("pageerror", lambda e: errors.append(str(e)))
            await page.goto(f"http://127.0.0.1:{server.server_port}/viewer.html")
            state = {
                "id": "failed-run",
                "status": "partial",
                "stage": "complete",
                "calls": 2,
                "attempt": 1,
                "created_at": 1,
                "completed_at": 10,
                "activity": [],
                "retry_decision": {
                    "stage": "refinement",
                    "message": "Gemini is temporarily unavailable (HTTP 503).",
                    "error_code": "PROVIDER_TRANSIENT",
                    "can_retry": True,
                    "retry_at": None,
                },
            }
            await page.evaluate(
                'state => {showProgress(state.stage,null,state);renderDecision("retryDecision",state)}',
                state,
            )
            assert "Needs your decision" == await page.locator("#progressTitle").inner_text()
            assert "HTTP 503" in await page.locator("#retryDecision").inner_text()
            assert await page.locator("#retryDecision input").input_value() == "1"

            async def approve(route):
                approvals.append(route.request.post_data_json)
                await route.fulfill(json={"job_id": "approved-child", "status": "queued"})

            await page.route("**/jobs/failed-run/retry", approve)
            await page.route(
                "**/jobs/approved-child",
                lambda route: route.fulfill(
                    json={
                        **state,
                        "id": "approved-child",
                        "status": "failed",
                        "stage": "failed",
                        "calls": 1,
                        "error_code": "PROVIDER_TRANSIENT",
                        "error_message": state["retry_decision"]["message"],
                    }
                ),
            )
            await page.wait_for_timeout(500)
            assert not approvals
            await page.get_by_role("button", name="Keep results — no more calls").click()
            assert not await page.locator("#retryDecision").is_visible()
            assert not approvals
            await page.evaluate('state => renderDecision("retryDecision",state)', state)
            await page.get_by_role("button", name="Approve retry", exact=True).click()
            await page.wait_for_function(
                "document.getElementById('runId').textContent.includes('approved-child')"
            )
            await page.wait_for_function("!document.getElementById('start').disabled")
            await page.wait_for_timeout(500)
            assert approvals == [{"approved": True, "max_model_calls": 1}]
            assert await page.locator("#retryDecision").is_visible()
            assert "No retry is running." in await page.locator("#retryDecision").inner_text()
            await page.evaluate(
                'state => showProgress("cancelled",null,{...state,status:"cancelled",retry_decision:null})',
                state,
            )
            assert "Cancelled by you" in await page.locator("#currentActivity").inner_text()
            assert not errors, errors
            await browser.close()
            print(
                json.dumps(
                    {
                        "retry_requires_click": "passed",
                        "single_call_approval": "passed",
                        "keep_results": "passed",
                        "cancellation_explanation": "passed",
                        "javascript_errors": errors,
                    }
                )
            )
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    asyncio.run(main())
