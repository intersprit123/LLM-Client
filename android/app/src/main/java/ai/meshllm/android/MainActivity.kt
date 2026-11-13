package ai.meshllm.android

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

private enum class ClusterMode { OFF, MASTER, WORKER }
private data class Peer(val id: String, val name: String, val detail: String, val score: Int)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MeshTheme { MeshApp() } }
    }
}

@Composable
private fun MeshTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = darkColorScheme(), content = content)
}

@Composable
private fun MeshApp() {
    var mode by remember { mutableStateOf(ClusterMode.OFF) }
    var autoBalance by remember { mutableStateOf(true) }
    var localOnly by remember { mutableStateOf(true) }
    var loaded by remember { mutableStateOf(false) }
    var running by remember { mutableStateOf(false) }
    var modelName by remember { mutableStateOf("Select a model") }
    var prompt by remember { mutableStateOf("") }
    var contextTokens by remember { mutableStateOf("8192") }
    var messages by remember { mutableStateOf(listOf<Pair<String, String>>()) }
    val peers = remember {
        listOf(
            Peer("p1", "This phone", "Android • local device", 100),
            Peer("p2", "Gaming laptop", "Windows • Wi-Fi • 8 GB VRAM", 82),
            Peer("p3", "Kali workstation", "Linux • Wi-Fi • 16 GB RAM", 61)
        )
    }
    val selected = remember { mutableStateListOf("p1") }
    val manualShares = remember { mutableStateMapOf("p1" to 50f, "p2" to 35f, "p3" to 15f) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Column {
                    Text("MeshLLM", style = MaterialTheme.typography.titleLarge)
                    Text("Distributed AI client • Android", style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant)
                } },
                actions = { IconButton(onClick = { }) { Icon(Icons.Default.Settings, "Settings") } }
            )
        },
        bottomBar = {
            NavigationBar {
                NavigationBarItem(selected = true, onClick = {}, icon = { Icon(Icons.Default.Hub, null) }, label = { Text("Cluster") })
                NavigationBarItem(selected = false, onClick = {}, icon = { Icon(Icons.Default.Search, null) }, label = { Text("Browse") })
                NavigationBarItem(selected = false, onClick = {}, icon = { Icon(Icons.Default.Folder, null) }, label = { Text("Models") })
                NavigationBarItem(selected = false, onClick = {}, icon = { Icon(Icons.Default.Chat, null) }, label = { Text("Chat") })
            }
        }
    ) { padding ->
        LazyColumn(
            modifier = Modifier.fillMaxSize().padding(padding),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            item {
                Card {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text("Connection mode", style = MaterialTheme.typography.titleMedium)
                                Text("Choose how this device participates", style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            AssistChip(onClick = {}, label = { Text("Wi-Fi only") }, leadingIcon = { Icon(Icons.Default.Wifi, null) })
                        }
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                            ModeButton("Off", mode == ClusterMode.OFF, Modifier.weight(1f)) { mode = ClusterMode.OFF; running = false }
                            ModeButton("Master", mode == ClusterMode.MASTER, Modifier.weight(1f)) { mode = ClusterMode.MASTER }
                            ModeButton("Worker", mode == ClusterMode.WORKER, Modifier.weight(1f)) { mode = ClusterMode.WORKER }
                        }
                        Text(
                            when (mode) {
                                ClusterMode.OFF -> "Disconnected. This device is not sharing resources."
                                ClusterMode.MASTER -> "Master controls model loading, starting, stopping and cluster settings."
                                ClusterMode.WORKER -> "Worker contributes only after the owner approves pairing and resource sharing."
                            },
                            style = MaterialTheme.typography.bodySmall
                        )
                        if (mode != ClusterMode.OFF) {
                            OutlinedButton(onClick = {}, modifier = Modifier.fillMaxWidth()) {
                                Icon(Icons.Default.QrCodeScanner, null)
                                Spacer(Modifier.width(8.dp))
                                Text(if (mode == ClusterMode.MASTER) "Show cluster QR / approve workers" else "Scan master's QR to join")
                            }
                        }
                    }
                }
            }

            item {
                SectionTitle("Connected devices", "Only approved devices should appear here")
                peers.forEach { peer ->
                    val checked = selected.contains(peer.id)
                    Card {
                        Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                            Checkbox(checked = checked, onCheckedChange = { yes ->
                                if (yes && peer.id !in selected) selected.add(peer.id)
                                if (!yes) selected.remove(peer.id)
                                if (selected.isEmpty()) localOnly = true
                            }, enabled = mode == ClusterMode.MASTER && peer.id != "p1")
                            Column(Modifier.weight(1f)) {
                                Text(peer.name, style = MaterialTheme.typography.titleSmall)
                                Text(peer.detail, style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            Text(if (peer.id == "p1") "MASTER" else "ONLINE", style = MaterialTheme.typography.labelSmall,
                                color = MaterialTheme.colorScheme.primary)
                        }
                    }
                }
                Text("Device list is a UI preview until Wi-Fi discovery and live coordinator sync are connected.",
                    style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }

            item {
                SectionTitle("Model", "Load first. Start second. Chat third.")
                Card {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        OutlinedTextField(value = modelName, onValueChange = { modelName = it },
                            label = { Text("Model name / local model path") }, modifier = Modifier.fillMaxWidth(), singleLine = true)
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("Use this device only", modifier = Modifier.weight(1f))
                            Switch(checked = localOnly, onCheckedChange = { localOnly = it })
                        }
                        Text(if (localOnly) "Local execution selected" else "${selected.size} selected device(s) available for planning",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = { loaded = modelName.isNotBlank(); running = false },
                                enabled = mode == ClusterMode.MASTER && modelName.isNotBlank(), modifier = Modifier.weight(1f)) {
                                Icon(Icons.Default.Download, null); Spacer(Modifier.width(6.dp)); Text("Load")
                            }
                            Button(onClick = { running = loaded }, enabled = loaded && mode == ClusterMode.MASTER,
                                modifier = Modifier.weight(1f)) {
                                Icon(Icons.Default.PlayArrow, null); Spacer(Modifier.width(6.dp)); Text("Start")
                            }
                        }
                        Text(if (running) "Running (prototype state)" else if (loaded) "Loaded — ready to start" else "No model loaded",
                            color = if (running) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                            style = MaterialTheme.typography.labelMedium)
                    }
                }
            }

            item {
                SectionTitle("Resource allocation", "Automatic is the default")
                Card {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text("Automatic allocation", style = MaterialTheme.typography.titleSmall)
                                Text("MeshLLM estimates safe shares from device capability and live headroom.",
                                    style = MaterialTheme.typography.bodySmall)
                            }
                            Switch(checked = autoBalance, onCheckedChange = { autoBalance = it })
                        }
                        if (autoBalance) {
                            peers.filter { it.id in selected && (it.id != "p1" || !localOnly) }.forEach { peer ->
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Text(peer.name, Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
                                    Text("Auto", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary)
                                }
                            }
                            Text("Automatic percentages will be calculated from measured throughput, available RAM/VRAM, thermal state, battery and Wi-Fi quality.",
                                style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        } else {
                            peers.filter { it.id in selected }.forEach { peer ->
                                Column {
                                    Row {
                                        Text(peer.name, Modifier.weight(1f))
                                        Text("${manualShares[peer.id]?.toInt() ?: 0}%")
                                    }
                                    Slider(value = manualShares[peer.id] ?: 0f, onValueChange = { manualShares[peer.id] = it })
                                }
                            }
                            Text("Manual shares are an operator target, not guaranteed compute capacity.",
                                style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }

            item {
                SectionTitle("Runtime settings", "Model-specific controls")
                Card {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(value = contextTokens, onValueChange = { contextTokens = it.filter(Char::isDigit) },
                            label = { Text("Context tokens") }, modifier = Modifier.fillMaxWidth(), singleLine = true)
                        Text("The final context/KV-cache budget must be checked against each device's memory and the selected runtime. MeshLLM will not split context blindly.",
                            style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }

            item {
                SectionTitle("Chat", "Available after a model is loaded and started")
                Card {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        if (messages.isEmpty()) Text("Your conversation will appear here.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                        messages.forEach { (who, msg) ->
                            Text("$who: $msg", style = MaterialTheme.typography.bodyMedium)
                        }
                        OutlinedTextField(value = prompt, onValueChange = { prompt = it },
                            label = { Text("Message the model") }, modifier = Modifier.fillMaxWidth(),
                            enabled = running, minLines = 2)
                        Button(onClick = {
                            if (prompt.isNotBlank()) {
                                messages = messages + ("You" to prompt)
                                prompt = ""
                            }
                        }, enabled = running && prompt.isNotBlank(), modifier = Modifier.fillMaxWidth()) {
                            Icon(Icons.Default.Send, null); Spacer(Modifier.width(8.dp)); Text("Send")
                        }
                    }
                }
            }
            item {
                Text("Android MVP scaffold • Controls currently demonstrate the intended flow; live QR pairing, Wi-Fi transport, model runtimes and inference are not connected yet.",
                    style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
private fun ModeButton(label: String, selected: Boolean, modifier: Modifier = Modifier, onClick: () -> Unit) {
    if (selected) Button(onClick = onClick, modifier = modifier) { Text(label) }
    else OutlinedButton(onClick = onClick, modifier = modifier) { Text(label) }
}

@Composable
private fun SectionTitle(title: String, subtitle: String) {
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text(title, style = MaterialTheme.typography.titleMedium)
        Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}
