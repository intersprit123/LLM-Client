package ai.meshllm.android

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URLEncoder
import java.net.URL

data class HubModel(
    val id: String,
    val downloads: Long,
    val likes: Long,
    val pipeline: String?,
    val library: String?,
    val tags: List<String>
)

/** Public Hugging Face model discovery. Results are metadata only, not trusted code. */
class HuggingFaceCatalog {
    suspend fun search(query: String, task: String? = null, limit: Int = 30): List<HubModel> =
        withContext(Dispatchers.IO) {
            val params = buildList {
                add("search=" + URLEncoder.encode(query, "UTF-8"))
                add("limit=" + limit.coerceIn(1, 100))
                add("sort=downloads")
                add("direction=-1")
                if (!task.isNullOrBlank()) add("pipeline_tag=" + URLEncoder.encode(task, "UTF-8"))
            }.joinToString("&")
            val connection = URL("https://huggingface.co/api/models?$params")
                .openConnection() as HttpURLConnection
            connection.connectTimeout = 10_000
            connection.readTimeout = 20_000
            connection.setRequestProperty("Accept", "application/json")
            try {
                if (connection.responseCode !in 200..299) error("Hugging Face HTTP ${connection.responseCode}")
                val array = JSONArray(connection.inputStream.bufferedReader().use { it.readText() })
                (0 until array.length()).map { i ->
                    val item = array.getJSONObject(i)
                    val tags = item.optJSONArray("tags") ?: JSONArray()
                    HubModel(
                        id = item.optString("modelId", item.optString("id")),
                        downloads = item.optLong("downloads"),
                        likes = item.optLong("likes"),
                        pipeline = item.optString("pipeline_tag").takeUnless { it == "null" || it.isBlank() },
                        library = item.optString("library_name").takeUnless { it == "null" || it.isBlank() },
                        tags = (0 until tags.length()).map { tags.optString(it) }
                    )
                }
            } finally { connection.disconnect() }
        }
}
