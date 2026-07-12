#!/usr/bin/env sh
set -eu

interval="${CRAWLER_INTERVAL_SECONDS:-21600}"
start_page="${CRAWLER_START_PAGE:-1}"
max_pages="${CRAWLER_MAX_PAGES:-4}"

case "$interval:$start_page:$max_pages" in
  *[!0-9:]*|:*|*::*|*:) echo "Crawler schedule values must be non-negative integers" >&2; exit 2 ;;
esac
if [ "$max_pages" -lt 1 ]; then
  echo "CRAWLER_MAX_PAGES must be at least 1" >&2
  exit 2
fi

run_crawler() {
  echo "==> Starting crawl: start_page=$start_page max_pages=$max_pages"
  scrapy crawl caoliu -a start_page="$start_page" -a max_page="$max_pages"
}

if [ "$interval" -eq 0 ]; then
  run_crawler
  exit 0
fi

while true; do
  if ! run_crawler; then
    echo "WARN: crawl failed; retrying on the next schedule" >&2
  fi
  echo "==> Next crawl in ${interval}s"
  sleep "$interval" &
  wait $!
done
