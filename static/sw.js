const CACHE_VERSION = 3;
const CACHE_NAME = `epl-predictor-v${CACHE_VERSION}`;

// Install: skip waiting to activate immediately
self.addEventListener('install', event => {
  self.skipWaiting();
});

// Activate: delete ALL old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch: only cache static assets. Never cache HTML or API responses.
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  if (url.origin !== location.origin) return;

  // HTML and API — always go to network, never cache
  if (event.request.mode === 'navigate'
    || event.request.destination === 'document'
    || url.pathname === '/'
    || url.pathname.startsWith('/api/')) {
    return;
  }

  // Static assets only — cache-first
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        if (cached) return cached;
        return fetch(event.request).then(response => {
          if (!response || response.status !== 200 || response.type !== 'basic') {
            return response;
          }
          const toCache = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, toCache));
          return response;
        });
      })
    );
  }
});
