from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
SPARKAI_URL = ''
SPARKAI_APP_ID = ''
SPARKAI_API_SECRET = ''
SPARKAI_API_KEY = ''
SPARKAI_DOMAIN = ''

def talk_to_spark(question):
    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=False,
    )
    messages = [ChatMessage(
        role="user",
        content=question
    )]
    handler = ChunkPrintHandler()
    ans = spark.generate([messages], callbacks=[handler])
    import re
    text = ans.generations[0][0].text.strip()
    text = re.sub(r'\n\s*\n+', '\n', text)  # 删除所有独立空行（包括包含空格的行）
    return text


if __name__ == '__main__':
    question = ""
    talk_to_spark(question)