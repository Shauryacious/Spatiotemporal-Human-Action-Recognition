"""Optional real-browser smoke: uv run --with playwright python scripts/smoke_viewer.py. Requires Chrome."""

import asyncio
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def main():
    with tempfile.TemporaryDirectory(prefix="shar-browser-") as temporary:
        env = {
            **os.environ,
            "SHAR_STORAGE_ROOT": temporary,
            "SHAR_PROVIDER": "mock",
            "SHAR_MODEL": "fixture-v1",
            "SHAR_AUTH_MODE": "local",
            "PYTHONPATH": str(ROOT / "src"),
            "PYDANTIC_DISABLE_PLUGINS": "__all__",
        }
        api = subprocess.Popen(
            [str(ROOT / ".venv/bin/shar"), "serve", "--port", "8765"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        worker = subprocess.Popen(
            [str(ROOT / ".venv/bin/shar"), "worker"],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for _ in range(1200):
                try:
                    urllib.request.urlopen("http://127.0.0.1:8765/ready", timeout=0.5)
                    break
                except Exception:
                    await asyncio.sleep(0.1)
            else:
                raise RuntimeError("Test server and worker did not become ready")
            async with async_playwright() as p:
                browser = await p.chromium.launch(channel="chrome", headless=True)
                page = await browser.new_page(viewport={"width": 1280, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                await page.goto("http://127.0.0.1:8765")
                await page.locator("#file").set_input_files(str(ROOT / "data/fixtures/cfr.mp4"))
                await page.wait_for_function("document.getElementById('video').videoWidth > 0")
                await page.locator("#start").click()
                assert "Enter an action query" in await page.locator("#inlineStatus").inner_text()
                assert await page.locator("#query").evaluate("el => el === document.activeElement")
                await page.locator('[data-view="settings"]').click()
                await page.locator("#testConnection").click()
                await page.wait_for_function(
                    "document.getElementById('settingsStatus').textContent.includes('Mock mode')"
                )
                await page.locator('[data-view="analyze"]').click()
                await page.locator("#query").fill("Find observable actions")
                await page.locator("#start").click()
                await page.wait_for_function(
                    "document.getElementById('status').textContent.startsWith('succeeded')",
                    timeout=30000,
                )
                assert await page.locator("#pipeline").is_visible()
                assert await page.locator("#progressTitle").inner_text() == "Analysis complete"
                assert await page.locator("#workProgress").evaluate("el => el.value === el.max")
                assert "Saving analysis results" in await page.locator("#activity").inner_text()
                assert await page.locator("#download").is_visible()
                assert "Mock provider" in await page.locator("#resultSummary").inner_text()
                assert not await page.locator("#spatial").is_checked()
                assert "Run ID:" in await page.locator("#runId").inner_text()
                await page.locator('[data-view="history"]').click()
                await page.locator("#historyList button").first.click()
                await page.wait_for_function(
                    "document.getElementById('historyData').textContent.includes('succeeded')"
                )
                assert "Find observable actions" in await page.locator("#historyData").inner_text()
                assert await page.locator("#historyDownload").is_visible()
                await page.reload()
                await page.locator('[data-view="history"]').click()
                await page.locator("#historyList button").first.click()
                await page.locator("#reviewRun").click()
                await page.wait_for_function(
                    "document.getElementById('view-analyze').classList.contains('active')"
                )
                assert "Mock provider" in await page.locator("#resultSummary").inner_text()
                await page.locator("#file").set_input_files(str(ROOT / "data/fixtures/cfr.mp4"))
                await page.wait_for_function("document.getElementById('video').videoWidth > 0")
                # Synthetic display fixture: never a claim about the uploaded footage.
                await page.evaluate("""() => {
                    report = {predictions:[{label:'waving',start_sec:0,end_sec:3,
                        confidence:.87,actor_ids:['actor-1'],spatial_status:'partial'}],
                        tracks:[{actor_id:'actor-1',samples:[{observed:true,timestamp_sec:1,
                        confidence:.91,bbox:{x:.1,y:.1,width:.3,height:.7}}]}],
                        video:{duration_sec:4},status:'succeeded',provenance:{provider:'fixture'}};
                    render();
                }""")
                assert "87.0%" in await page.locator("#actions").inner_text()
                assert "00:00.00 – 00:03.00" in await page.locator("#actions").inner_text()
                await page.get_by_role("button", name="Show first observed box").click()
                await page.wait_for_function(
                    "document.getElementById('playbackStatus').textContent.includes('1 observed box')"
                )
                assert await page.locator("#overlay").evaluate(
                    "c => c.getContext('2d').getImageData(0,0,c.width,c.height).data.some(v => v > 0)"
                )
                await page.evaluate("document.getElementById('video').currentTime = 2")
                await page.wait_for_function(
                    "document.getElementById('playbackStatus').textContent.includes('0 observed box')"
                )
                import io

                from PIL import Image

                from shar.config import Settings
                from shar.storage.artifacts import LocalArtifacts
                from shar.storage.repository import Repository

                repository = Repository(Settings(storage_root=Path(temporary)))
                session = repository.start_stream("local", "synthetic-browser-check")
                repository.append_stream_event(
                    session,
                    "local",
                    {"type": "event", "revision": 1, "label": "<img src=x onerror=alert(1)>"},
                )
                preview = io.BytesIO()
                Image.new("RGB", (64, 48), "navy").save(preview, format="JPEG")
                LocalArtifacts(Path(temporary)).put(
                    f"streams/{session}/preview.jpg", preview.getvalue()
                )
                await page.locator('[data-view="stream"]').click()
                await page.locator("#streamId").fill(session)
                await page.locator("#monitor").click()
                await page.wait_for_function(
                    "document.getElementById('streamPreview').naturalWidth > 0"
                )
                assert "revision" in await page.locator("#streamEvents").inner_text()
                assert not errors, errors
                # Exercise the polling UI with a terminal quota failure. No live provider calls.
                await page.route(
                    "**/jobs/quota-fixture",
                    lambda route: route.fulfill(
                        json={
                            "id": "quota-fixture",
                            "status": "failed",
                            "stage": "failed",
                            "calls": 1,
                            "attempt": 1,
                            "created_at": 1700000000,
                            "completed_at": 1700000002,
                            "error_code": "PROVIDER_DAILY_QUOTA",
                            "error_message": "Gemini's daily quota is exhausted. Check Google AI Studio.",
                            "retry_at": 1700053000,
                            "activity": [
                                {
                                    "at": 1700000001,
                                    "event": "checkpoint",
                                    "details": {
                                        "stage": "analysis",
                                        "message": "Sampling window 1/1",
                                    },
                                }
                            ],
                        }
                    ),
                )
                await page.locator('[data-view="analyze"]').click()
                await page.evaluate(
                    "async () => {clearResults(); job='quota-fixture'; await refresh();}"
                )
                assert "daily quota" in await page.locator("#currentActivity").inner_text()
                assert "Try again after" in await page.locator("#currentActivity").inner_text()
                assert await page.locator('#checkpoints li[data-state="failed"]').count() == 1
                assert await page.locator('#checkpoints li[data-state="done"]').count() == 2
                await page.get_by_text("Analysis diagnostics", exact=True).click()
                assert "PROVIDER_DAILY_QUOTA" in await page.locator("#details").inner_text()
                assert await page.locator("#start").is_enabled()
                assert not errors, errors
                uploads = []
                page.on(
                    "request",
                    lambda request: (
                        uploads.append(request.url)
                        if request.method == "POST" and request.url.endswith("/videos")
                        else None
                    ),
                )
                await page.route(
                    "**/connection",
                    lambda route: route.fulfill(
                        status=429,
                        json={
                            "error": {
                                "code": "PROVIDER_DAILY_QUOTA",
                                "message": "Gemini's daily quota is exhausted.",
                                "retry_at": 1700053000,
                            },
                        },
                    ),
                )
                await page.locator("#start").click()
                await page.wait_for_function(
                    "document.getElementById('currentActivity').textContent.includes('daily quota')"
                )
                assert not uploads, uploads
                assert "Try again after" in await page.locator("#currentActivity").inner_text()
                assert await page.locator("#start").is_enabled()
                await page.locator('[data-view="settings"]').click()
                await page.locator("#recheckProvider").click()
                await page.wait_for_function(
                    "document.getElementById('settingsStatus').textContent.includes('pause cleared')"
                )
                await page.locator('[data-view="analyze"]').click()
                await page.screenshot(path=str(ROOT / "outputs/viewer-smoke.png"), full_page=True)
                print(
                    {
                        "viewer_upload_worker_result": "passed",
                        "stream_polling_preview": "passed",
                        "quota_failure_explanation": "passed",
                        "quota_preflight_blocks_upload": "passed",
                        "javascript_errors": errors,
                        "status": await page.locator("#status").inner_text(),
                    }
                )
                await browser.close()
        finally:
            for process in (worker, api):
                process.terminate()
            for process in (worker, api):
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    asyncio.run(main())
