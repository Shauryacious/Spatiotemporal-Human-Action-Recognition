# Free API setup

This application currently supports Gemini for real inference and a mock provider for
tests. Creating an API key does not itself grant additional inference quota.

## Gemini: works with this application

1. Sign in to [Google AI Studio](https://aistudio.google.com/apikey).
2. Select **Create API key** and select a project you own. For new users, AI Studio
   creates a default project; existing Cloud projects can be imported through Projects.
3. Check the project's available models and free-tier limits in
   [Usage / Rate limits](https://ai.dev/rate-limit).
4. Replace `GEMINI_API_KEY` in your local `.env` file. Keep your app's bearer token
   (`SHAR_AUTH_TOKENS`) separate. Set `SHAR_MODEL` to an accessible vision model and
   set `SHAR_RPM`, `SHAR_TPM`, and `SHAR_RPD` to your actual limits.
5. For the managed local services in this workspace, restart both with:

   ```sh
   python3 scripts/restart_local.py
   ```

   This reloads `.env`, uses the repository's `runtime` directory, and keeps logs in
   `runtime/logs`. Finish or cancel active jobs first. Refresh the browser afterward.

   If you instead run the services manually, stop their existing processes and run:

   ```sh
   # API terminal
   uv run --env-file .env shar serve

   # Worker terminal
   uv run --env-file .env shar worker
   ```

Google applies quotas per project, not per API key. Replacing a key in an exhausted
project does not reset its quota. Daily limits reset at midnight Pacific time; the app
displays the provider's retry time in your browser's timezone. Another supported model
may have available quota; check AI Studio before changing `SHAR_MODEL`.

For a basic analysis with fewer calls, turn off **Localize actors** and **Refine
boundaries**. Longer videos still need multiple windows; retries also consume calls.

### Call budgets and retry approval

Interactive runs stop after the first failed model call, including invalid model
output. There are no automatic provider retries or automatic output-correction calls.
The failed stage, reason, provider retry time (when supplied), and used call count are
shown in Analyze and History. Successful results remain available for review.

The viewer defaults to at most three new calls per run. Recognition uses one call per
video window. Refine boundaries uses at most one call per detected action; it no longer
runs both iterative boundary search and a second refinement procedure. Actor
localization uses at most one call per action. The new-call limit also applies to
long videos or videos with several actions, which may need additional user approval.

Choose **Approve retry** to authorize a new run linked to the original, or **Keep
results — no more calls**. The retry budget defaults to one new call. Successful
responses are stored with the run and reused even after the ordinary cache expires.
A missing or damaged saved response stops the retry rather than silently regenerating
it. Duplicate approvals create only one child run. A worker interruption also requires
approval because an in-flight request might already have been charged.

`SHAR_REQUIRE_RETRY_APPROVAL=true` is the default and is always enforced by the web
submission and retry endpoints. The lower-level pipeline retains explicit opt-in
unattended retry settings for programmatic use; these are not used by the viewer.
The server's `SHAR_MAX_CALLS` remains an upper bound on user-selected new-call budgets.
Restart both services after changing `.env`. Saved responses follow video/run
retention, so expired runs cannot be resumed.

Official references: [API keys](https://ai.google.dev/gemini-api/docs/api-key),
[rate limits](https://ai.google.dev/gemini-api/docs/rate-limits),
[free and paid pricing](https://ai.google.dev/gemini-api/docs/pricing).

## Other services with limited free access

- **Groq:** Sign up at [Groq Console](https://console.groq.com/keys), then create an API
  key. See [free-plan limits](https://console.groq.com/docs/rate-limits) and
  [vision models](https://console.groq.com/docs/vision). Image limits vary by model.
- **OpenRouter:** Sign up and create a key at
  [Settings → API Keys](https://openrouter.ai/settings/keys). Choose a free model that
  accepts images, and check [account limits](https://openrouter.ai/docs/api_reference/limits).
  A free key does not make paid models free.

These services require additional adapters in this repository. Their keys cannot be
substituted into `GEMINI_API_KEY`. Free tiers have account/model limits and availability
can change; creating extra keys does not guarantee additional capacity.
