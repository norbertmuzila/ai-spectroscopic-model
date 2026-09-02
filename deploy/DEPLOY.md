# Deployment

## The constraint, first

**A hosted web app cannot read your spectrometer.** The USB4000 is a USB device
on your desk; a container in a datacentre has no USB bus, and a browser tab
cannot open one either. This is an operating-system and browser-security
boundary, not a configuration problem, and no hosting provider changes it.

So there are two deployments, and they do different jobs:

| | What it does | Where it runs |
|---|---|---|
| **Instrument console** | Reads the USB4000, takes references, streams live, writes PDF reports | Your machine — `START.bat` or `python run.py` |
| **Hosted demo** | The same dashboard and engine, driven by the built-in simulator | Free cloud host, permanent public URL |

If you want a **public URL that reaches the real hardware**, that is option C
below: the app stays on your machine and a tunnel publishes it.

---

## A. Hugging Face Spaces — free, no expiry, WebSockets work

The best free option for this app: 2 vCPU / 16 GB RAM, no card required, no
trial period, and it supports the WebSocket the live view needs. A Space sleeps
after prolonged inactivity and wakes on the next request.

1. Create an account at <https://huggingface.co/join>.
2. <https://huggingface.co/new-space> → name it `spectral-console`,
   **SDK: Docker**, hardware **CPU basic (free)**, visibility Public.
3. Push this project to it:

   ```bash
   cp deploy/space-README.md README-space.md      # keep your own README
   git init
   git add -A
   git commit -m "USB4000 spectral console"
   git remote add space https://huggingface.co/spaces/<your-username>/spectral-console
   git push space main
   ```

   In the Space repo the file named `README.md` must be the one with the YAML
   frontmatter — copy `deploy/space-README.md` over it there.
4. The Space builds the Docker image (it trains the model during the build, so
   the first build takes 10–15 minutes) and then serves at
   `https://huggingface.co/spaces/<your-username>/spectral-console`.

## B. Render — free, no expiry, blueprint included

1. Push the project to GitHub (`gh repo create` or the web UI).
2. <https://render.com> → **New → Blueprint** → select the repository.
   `deploy/render.yaml` is picked up automatically.

The free instance sleeps after ~15 minutes idle and cold-starts in under a
minute. It does not expire.

### A note on Vercel

Vercel's free tier is genuinely permanent, but it is a poor fit here and I would
not recommend it: its Python runtime is serverless, so there is no long-lived
process to hold the WebSocket the live view uses, no writable disk for the
SQLite history, and the 250 MB bundle limit is tight against scikit-learn plus
the 52 MB model. It can be made to work by stripping live streaming and history,
but the result is a worse demo than either option above.

## C. A public URL that *does* reach your spectrometer

Run the app locally as normal, then publish that local port with a tunnel. This
is the only way to get a public link that reads real hardware, because the
measurement still happens on your machine.

```bash
winget install --id Cloudflare.cloudflared
python run.py                       # in one terminal
cloudflared tunnel --url http://127.0.0.1:8000    # in another
```

`cloudflared` prints a `https://<random>.trycloudflare.com` URL that anyone can
open. It lives as long as the command runs. For a **stable** address that
survives restarts, create a free named tunnel:

```bash
cloudflared tunnel login
cloudflared tunnel create spectro
cloudflared tunnel route dns spectro spectro.<your-domain>
cloudflared tunnel run --url http://127.0.0.1:8000 spectro
```

A named tunnel needs a domain on a free Cloudflare account; the quick tunnel
above needs nothing at all.

**Before you expose it:** the API has no authentication — anyone with the link
can trigger measurements and read your stored analyses. Use the quick tunnel for
short demos, and put Cloudflare Access in front of a named tunnel if it is going
to stay up.

---

## Local build check

```bash
docker build -t spectral-console .
docker run -p 8000:8000 spectral-console
```

Docker is not currently installed on this machine, so the image is written and
reviewed but has not been built here. Both hosts above build it for you, so a
local Docker install is only needed if you want to test the image yourself.
