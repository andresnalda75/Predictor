const CACHE_VERSION = 2;
const CACHE_NAME = `epl-predictor-v${CACHE_VERSION}`;

const PRECACHE_URLS = [
  '/static/manifest.json',
];

// Install: precache static shell
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(PRECACHE_URLS))
  );
  self.skipWaiting();
});

// Activate: remove old caches (busts cache on version bump)
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch strategy:
//   - Network-first for HTML pages and API calls (always fresh)
//   - Cache-first for static assets (images, CSS, JS, fonts)
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Never intercept cross-origin requests (fonts, crests, etc.)
  if (url.origin !== location.origin) return;

  const isHTML = event.request.mode === 'navigate'
    || event.request.destination === 'document'
    || url.pathname === '/';
  const isAPI = url.pathname.startsWith('/api/');

  // Network-first for HTML pages and API routes
  if (isHTML || isAPI) {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          if (response && response.status === 200 && response.type === 'basic') {
            const toCache = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(event.request, toCache));
          }
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // Cache-first for static assets (/static/*)
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
});
