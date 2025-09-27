# MLSysOps-Hackaton

Perfetto, hai una lista piuttosto completa di metriche che arrivano da **Prometheus** e da vari **exporter** (Go runtime, Node Exporter, Kubernetes, applicativi custom).
Ti faccio una panoramica divisa per categoria così è più chiaro cosa misurano:

---

## 🔹 Metriche applicative custom (legate a video/frames/classificazione)

* **box_count** → Numero di “box” rilevati (probabile oggetti in bounding box da un detector).
* **classifier_frames_processed_total** → Totale di frame elaborati dal classificatore.
* **detector_disconnected** → Flag (0/1) che indica se il detector è disconnesso.
* **detector_disconnected_seconds_total** → Tempo cumulativo in secondi in cui il detector è stato disconnesso.
* **detector_queue_size** → Dimensione della coda di frame in attesa di essere processati.
* **frame_classify_latency** → Latenza della classificazione di un frame.
* **frame_duration** → Durata di un frame (probabilmente in millisecondi).
* **frame_latency** → Tempo che un frame impiega tra acquisizione e elaborazione.
* **frames_sent_total** → Numero totale di frame inviati.
* **object_classify_latency** → Latenza nella classificazione di un oggetto (non di un frame intero).
* **video_index** → Indice/contatore dei video processati.

---

## 🔹 Go runtime (esposte da `prometheus/client_golang`)

* **go_gc_duration_seconds**, **_count**, **_sum** → Durata delle pause di garbage collection (GC).
* **go_gc_gogc_percent** → Percentuale di GOGC impostata (quando scatta il GC).
* **go_gc_gomemlimit_bytes** → Limite di memoria Go per il GC.
* **go_goroutines** → Numero di goroutine attive.
* **go_info** → Info versione runtime Go.
* **go_memstats_*** → Statistiche di memoria Go (heap, stack, allocazioni, GC, ecc.).
* **go_sched_gomaxprocs_threads** → Numero massimo di thread schedulabili.
* **go_threads** → Numero di thread OS creati.

---

## 🔹 Kubernetes metrics (kube-state-metrics)

* **k8s_container_cpu_limit / request** → Limiti e richieste CPU del container.
* **k8s_container_memory_limit_bytes / request_bytes** → Limiti e richieste memoria container.
* **k8s_container_ready** → Stato readiness container (0/1).
* **k8s_container_restarts** → Numero di restart del container.
* **k8s_daemonset_* , k8s_deployment_*, k8s_replicaset_*, k8s_statefulset_*** → Stato di replica/rollout dei workload Kubernetes.
* **k8s_job_* (active_pods, failed_pods, successful_pods, ecc.)** → Stato dei job.
* **k8s_namespace_phase** → Stato del namespace (Active/Terminating).
* **k8s_node_allocatable_cpu / memory_bytes** → Risorse allocabili su un nodo.
* **k8s_node_condition_ready / memory_pressure** → Stato del nodo (Ready, MemoryPressure).
* **k8s_pod_phase** → Stato dei pod (Running, Pending, Failed).

---

## 🔹 Node Exporter (sistema host)

* **node_cpu_seconds_total / node_cpu_guest_seconds_total** → Tempo CPU speso in vari stati (user, system, idle, guest).
* **node_exporter_build_info** → Info versione Node Exporter.
* **node_memory_*** → Metriche di memoria (MemTotal, MemFree, Cached, HugePages, Slab, Swap, ecc.).
* **node_os_info / node_os_version** → Info sul sistema operativo.
* **node_scrape_collector_duration_seconds / success** → Tempo e successo della raccolta di metriche da ogni collector.

---

## 🔹 Process exporter (processi applicativi)

* **process_cpu_seconds_total** → Tempo CPU totale consumato dal processo.
* **process_max_fds** → Numero massimo di file descriptor.
* **process_open_fds** → Numero di file descriptor attualmente aperti.
* **process_network_receive_bytes_total / transmit_bytes_total** → Traffico rete del processo.
* **process_resident_memory_bytes** → Memoria residente usata dal processo.
* **process_virtual_memory_bytes / max_bytes** → Memoria virtuale usata e massima disponibile.
* **process_start_time_seconds** → Timestamp di avvio del processo.

---

## 🔹 Prometheus stesso

* **promhttp_metric_handler_errors_total** → Errori nel servire le metriche.
* **promhttp_metric_handler_requests_in_flight** → Richieste HTTP in corso al server di metriche.
* **promhttp_metric_handler_requests_total** → Totale richieste HTTP alle metriche.
* **scrape_duration_seconds** → Durata di uno scrape.
* **scrape_samples_scraped** → Numero di campioni raccolti.
* **scrape_samples_post_metric_relabeling** → Campioni rimasti dopo il relabeling.
* **scrape_series_added** → Nuove serie aggiunte.
* **target_info** → Info sul target.
* **up** → Stato del target (1 se risponde, 0 se down).

---

👉 In sintesi:

* Le prime metriche sono **specifiche del tuo applicativo (video/frames/classificazione)**.
* Le metriche **Go e process** descrivono lo stato del runtime e dei processi.
* Le metriche **Kubernetes** descrivono lo stato dei container, pod e risorse.
* Le metriche **Node Exporter** mostrano lo stato del sistema operativo/nodo.
* Le metriche **Prometheus** monitorano il funzionamento della raccolta stessa.

---

Vuoi che ti prepari una **tabella riassuntiva con nome metrica → spiegazione** (tipo glossario), così puoi usarla direttamente come riferimento rapido?

