# Define here the models for your scraped items
import scrapy


class CaoliuItem(scrapy.Item):
    url = scrapy.Field()
    video_id = scrapy.Field()  # legacy local folder id
    content_key = scrapy.Field()
    title = scrapy.Field()
    title_raw = scrapy.Field()
    title_clean = scrapy.Field()
    image_urls = scrapy.Field()
    images = scrapy.Field()
    download_link = scrapy.Field()
    magnet_uri = scrapy.Field()
    info_hash = scrapy.Field()
    source_url = scrapy.Field()
