package ai.meshllm.android

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * Small REST client for the existing MeshLLM coordinator.
 * Use only a trusted LAN endpoint until TLS is configured.
 */
class MeshApi(private val baseUrl: String) {
    private fun request(path: String, method: String = "GET", body: JSONObject? = null,
                        headers: Map<String, String> = emptyMap()): String {
        val url = URL(baseUrl.trimEnd('/') + path)
        val connection = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 8_000
            readTimeout = 20_000
            headers.forEach { (key, value) -> setRequestProperty(key, value) }
            if (body != null) {
                doOutput = true
                setRequestProperty("Content-Type", "application/json")
            }
        }
        try {
            if (body != null) connection.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val payload = stream?.bufferedReader()?.use { it.readText() }.orEmpty()
            if (code !in 200..299) throw MeshApiException(code, payload)
            return payload
        } finally {
            connection.disconnect()
        }
    }

    suspend fun health(): JSONObject = io { JSONObject(request("/api/health")) }
    suspend fun workers(): JSONArray = io { JSONArray(request("/api/workers")) }
    suspend fun schedulerPreview(): JSONObject = io { JSONObject(request("/api/scheduler/preview")) }

    suspend fun createPairingSession(metadata: JSONObject): JSONObject =
        io { JSONObject(request("/api/pairing-sessions", "POST", metadata)) }

    suspend fun approvePairing(sessionId: String, workerSecret: String): JSONObject =
        io {
            request("/api/pairing-sessions/$sessionId/approve", "POST",
                JSONObject().put("worker_secret", workerSecret))
            JSONObject().put("approved", true)
        }

    suspend fun submitPrompt(model: String, prompt: String, maxTokens: Int = 512): JSONObject =
        io {
            JSONObject(request("/api/jobs", "POST", JSONObject()
                .put("model", model).put("prompt", prompt).put("max_tokens", maxTokens)))
        }

    suspend fun job(jobId: String): JSONObject =
        io { JSONObject(request("/api/jobs/$jobId")) }

    suspend fun leave(workerId: String, workerToken: String): JSONObject =
        io { JSONObject(request("/api/workers/$workerId/leave", "POST",
            headers = mapOf("X-Worker-Token" to workerToken))) }

    private suspend fun <T> io(block: () -> T): T = withContext(Dispatchers.IO) { block() }
}

class MeshApiException(val status: Int, detail: String) :
    Exception("MeshLLM coordinator returned HTTP $status: $detail")
