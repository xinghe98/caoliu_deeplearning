# Scrapy settings for caoliu project
import os
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

LOG_LEVEL = "DEBUG"
BOT_NAME = "caoliu"

SPIDER_MODULES = ["caoliu.spiders"]
NEWSPIDER_MODULE = "caoliu.spiders"

ADDONS = {}


# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# Concurrency and throttling settings
# CONCURRENT_REQUESTS = 16
CONCURRENT_REQUESTS_PER_DOMAIN = 1
DOWNLOAD_DELAY = 1

# Disable cookies (enabled by default)
# COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
# TELNETCONSOLE_ENABLED = False

# Override the default request headers:
# DEFAULT_REQUEST_HEADERS = {
#    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#    "Accept-Language": "en",
# }

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
# SPIDER_MIDDLEWARES = {
#    "caoliu.middlewares.CaoliuSpiderMiddleware": 543,
# }

# Keep OffsiteMiddleware; external image requests use dont_filter=True in pipeline.
DOWNLOADER_MIDDLEWARES = {}
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 1.0
AUTOTHROTTLE_MAX_DELAY = 10.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
# EXTENSIONS = {
#    "scrapy.extensions.telnet.TelnetConsole": None,
# }

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    "caoliu.pipelines.CaoliuIndexPipeline": 1,  # stable content_key + CSV audit
    "caoliu.pipelines.CaoliuImagesPipeline": 100,  # download + validate images
    "caoliu.pipelines.PlatformIngestPipeline": 250,  # platform ingest after images
    "caoliu.pipelines.CaoliuPipeline": 300,
}

# Platform ingest. Keep credentials outside Git and override through environment variables.
PLATFORM_API_URL = os.environ.get("PLATFORM_API_URL", "")
INGEST_API_KEY = os.environ.get("INGEST_API_KEY", "")
PLATFORM_INGEST_REQUIRED = os.environ.get("PLATFORM_INGEST_REQUIRED", "false").lower() in {"1", "true", "yes"}
PLATFORM_INGEST_RETRIES = int(os.environ.get("PLATFORM_INGEST_RETRIES", "3"))
PLATFORM_INGEST_TIMEOUT = float(os.environ.get("PLATFORM_INGEST_TIMEOUT", "30"))

# ============ 草榴爬虫配置 ============
# 下载根目录 (可自定义修改)
CAOLIU_DOWNLOAD_DIR = os.environ.get("CAOLIU_DOWNLOAD_DIR", "../downloads")

# 图片保存路径 (相对于项目根目录)
IMAGES_STORE = os.environ.get("IMAGES_STORE", CAOLIU_DOWNLOAD_DIR)

# 图片下载超时
IMAGES_DOWNLOAD_TIMEOUT = 30

# 允许重定向
MEDIA_ALLOW_REDIRECTS = True

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
# AUTOTHROTTLE_ENABLED = True
# The initial download delay
# AUTOTHROTTLE_START_DELAY = 5
# The maximum download delay to be set in case of high latencies
# AUTOTHROTTLE_MAX_DELAY = 60
# The average number of requests Scrapy should be sending in parallel to
# each remote server
# AUTOTHROTTLE_TARGET_CONCURRENCY = 1.0
# Enable showing throttling stats for every response received:
# AUTOTHROTTLE_DEBUG = False

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
# HTTPCACHE_ENABLED = True
# HTTPCACHE_EXPIRATION_SECS = 0
# HTTPCACHE_DIR = "httpcache"
# HTTPCACHE_IGNORE_HTTP_CODES = []
# HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Set settings whose default value is deprecated to a future-proof value
FEED_EXPORT_ENCODING = "utf-8"
