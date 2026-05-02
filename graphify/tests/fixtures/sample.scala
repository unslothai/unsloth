import scala.collection.mutable.ListBuffer

case class Config(baseUrl: String, timeout: Int)

class HttpClient(config: Config) {
  def get(path: String): String = {
    buildRequest("GET", path)
  }

  def post(path: String, body: String): String = {
    buildRequest("POST", path)
  }

  private def buildRequest(method: String, path: String): String = {
    s"$method ${config.baseUrl}$path"
  }
}

object HttpClientFactory {
  def create(baseUrl: String): HttpClient = {
    new HttpClient(Config(baseUrl, 30))
  }
}
