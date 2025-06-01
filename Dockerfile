# Dockerfile

# =============================================
# Builder Stage
# =============================================
FROM python:3.10-slim-bookworm AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libatk1.0-0 libatk-bridge2.0-0 libcups2 libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 libnspr4 libnss3 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 libxrender1 libxtst6 ca-certificates fonts-liberation libasound2 libpangocairo-1.0-0 libpango-1.0-0 libu2f-udev \
    supervisor curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app_builder
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -U "camoufox[geoip]" && \
    pip install --no-cache-dir -r requirements.txt && \
    camoufox fetch

# =============================================
# Final Stage
# =============================================
FROM python:3.10-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libatk1.0-0 libatk-bridge2.0-0 libcups2 libdbus-1-3 libdrm2 libgbm1 libgtk-3-0 libnspr4 libnss3 libx11-6 libx11-xcb1 libxcb1 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 libxrender1 libxtst6 ca-certificates fonts-liberation libasound2 libpangocairo-1.0-0 libpango-1.0-0 libu2f-udev \
    supervisor curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

RUN python -m playwright install-deps firefox

COPY . .

# Create user, group, and set up initial directories and permissions AS ROOT
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -s /bin/bash -d /app appuser && \
    mkdir -p /app/.cache/camoufox
# Copy the camoufox cache from the builder stage
COPY --from=builder /root/.cache/camoufox/ /app/.cache/camoufox/

RUN \
    # Create /home/appuser AS ROOT and give ownership to appuser
    # This is crucial for PLAYWRIGHT_BROWSERS_PATH
    mkdir -p /home/appuser/.cache/ms-playwright && \
    mkdir -p /home/appuser/.mozilla && \
    chown -R appuser:appgroup /app && \
    chown -R appuser:appgroup /home/appuser
# Switch to non-root user
USER appuser
ENV HOME=/app

# Install Playwright browser as appuser
RUN python -m playwright install firefox

# Create runtime directories needed by the application (as appuser)
# These will now be created inside /app or /home/appuser which appuser owns
RUN mkdir -p /app/logs && \
    mkdir -p /app/auth_profiles/active && \
    mkdir -p /app/auth_profiles/saved && \
    mkdir -p /app/certs
    # The /home/appuser/... directories are already created by root and chowned,
    # or if appuser needs to create deeper subdirs, it now has permission in /home/appuser.

COPY supervisord.conf /etc/supervisor/conf.d/app.conf

EXPOSE 2048
EXPOSE 3120

ENV PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright
ENV PYTHONUNBUFFERED=1
ENV SERVER_PORT=2048
ENV STREAM_PORT=3120
ENV INTERNAL_CAMOUFOX_PROXY=""

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/app.conf"]