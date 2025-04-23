import polars as pl
import numpy as np
import datetime
import random
import uuid

def generate_supercomputer_logs(num_logs=100, anomaly_ratio=0.1, log_type="mixed"):
    """Generate synthetic supercomputer logs for testing the pipeline"""
    print(f"Generating {num_logs} synthetic {log_type} supercomputer logs...")
    
    logs = []
    timestamps = []
    node_ids = []
    levels = []
    components = []
    thread_ids = []
    
    base_time = datetime.datetime.now() - datetime.timedelta(hours=24)
    
    possible_nodes = [f"node-{i:04d}" for i in range(1, 51)]
    
    hadoop_components = [
        "org.apache.hadoop.hdfs.server.namenode.NameNode",
        "org.apache.hadoop.hdfs.server.datanode.DataNode",
        "org.apache.hadoop.yarn.server.resourcemanager.ResourceManager",
        "org.apache.hadoop.yarn.server.nodemanager.NodeManager",
        "org.apache.hadoop.mapreduce.JobTracker",
        "org.apache.hadoop.mapreduce.TaskTracker"
    ]
    
    bgl_components = ["KERNEL", "MEMORY", "NETWORK", "APP", "IO", "SCHEDULER", "MPI", "HARDWARE", "FILESYSTEM"]
    
    thread_template = "Thread-{}"
    
    hadoop_templates = [
        "INFO: Started {service} at {address} with allocated memory: {memory}MB",
        "INFO: Block {block} replicated to {num_nodes} nodes",
        "INFO: Map {map_id} completed successfully for job {job_id}",
        "WARN: High GC overhead detected, GC took {gc_time}ms",
        "ERROR: Failed to write block {block} to {path}, {error}",
        "INFO: Container {container} launched on {host}",
        "WARN: Slow ReadProcessor read took {read_time}ms",
        "ERROR: Exception in heartbeat from node {node}: {error}",
        "INFO: Reducer {reducer_id} 100% complete for job {job_id}",
        "ERROR: FSDirectory lock acquisition failed: {error}"
    ]
    
    bgl_templates = [
        "FATAL: Machine check interrupt on CPU {cpu}, core {core}",
        "ERROR: ECC error detected in memory module {module}",
        "WARNING: Link card failure detected on node {node_id}, port {port}",
        "INFO: Job {job_id} started on {num_nodes} nodes",
        "ERROR: Timeout waiting for barrier synchronization on rank {rank}",
        "WARNING: Temperature threshold exceeded on node {node_id}: {temp}°C",
        "ERROR: MPI communication error: {error} on rank {rank}",
        "FATAL: Uncorrectable memory error detected at address {address}",
        "WARNING: Network congestion detected on switch {switch}, utilization {util}%",
        "ERROR: I/O node {io_node} not responding, {reason}"
    ]
    
    # Choose templates based on log type
    templates = hadoop_templates + bgl_templates if log_type == "mixed" else \
                hadoop_templates if log_type == "hadoop" else bgl_templates
    component_list = hadoop_components + bgl_components if log_type == "mixed" else \
                     hadoop_components if log_type == "hadoop" else bgl_components
    
    # Add some job IDs
    job_ids = [f"job_{uuid.uuid4().hex[:8]}" for _ in range(10)]
    
    # Common error messages
    errors = [
        "Connection refused", "Operation timed out", "No route to host",
        "Out of memory", "Permission denied", "Resource temporarily unavailable",
        "Broken pipe", "Input/output error", "Too many open files"
    ]
    
    # Generate logs
    for i in range(num_logs):
        # Generate timestamp with some clustering
        cluster_offset = (i // 5) * 600 + random.randint(0, 30)
        timestamp = base_time + datetime.timedelta(seconds=cluster_offset)
        timestamps.append(timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        
        # Select a node - use some locality
        node_id = possible_nodes[i // 3 % len(possible_nodes)]
        node_ids.append(node_id)
        
        # Select component/class
        component = random.choice(component_list)
        components.append(component)
        
        # Generate thread ID
        thread_id = thread_template.format(random.randint(1, 100))
        thread_ids.append(thread_id)
        
        # Select template
        template = random.choice(templates)
        
        # Placeholder values
        values = {
            "service": random.choice(["NameNode", "DataNode", "ResourceManager", "NodeManager"]),
            "address": f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}:8020",
            "memory": random.randint(1024, 8192),
            "block": f"blk_{uuid.uuid4().hex[:16]}",
            "num_nodes": random.randint(1, 10),
            "job_id": random.choice(job_ids),
            "map_id": f"map_{random.randint(0, 9999):04d}",
            "gc_time": random.randint(500, 10000),
            "path": f"/data/hadoop/tmp/dfs/data/{random.randint(1, 999)}",
            "error": random.choice(errors),
            "container": f"container_{uuid.uuid4().hex[:8]}",
            "host": f"host-{random.randint(1, 50):02d}.cluster.local",
            "read_time": random.randint(200, 5000),
            "node": node_id,
            "reducer_id": f"reduce_{random.randint(0, 9999):04d}",
            "cpu": random.randint(0, 63),
            "core": random.randint(0, 23),
            "module": f"DIMM_{random.choice(['A', 'B', 'C', 'D'])}{random.randint(0, 7)}",
            "port": random.randint(0, 7),
            "rank": random.randint(0, 1023),
            "temp": random.randint(85, 105),
            "switch": f"switch-{random.randint(1, 24):02d}",
            "util": random.randint(85, 100),
            "io_node": f"io-{random.randint(1, 8):02d}",
            "reason": random.choice(["timeout", "hardware failure", "connection lost", "filesystem error"])
        }
        
        # Replace placeholders
        log_message = template
        for key, value in values.items():
            if "{" + key + "}" in log_message:
                log_message = log_message.replace("{" + key + "}", str(value))
        
        # Extract log level from template
        if "INFO:" in template:
            level = "INFO"
        elif "WARN" in template or "WARNING" in template:
            level = "WARNING"
        elif "ERROR" in template:
            level = "ERROR"
        elif "FATAL" in template:
            level = "FATAL"
        else:
            level = "INFO"
        
        levels.append(level)
        logs.append(log_message)
    
    # Create DataFrame
    df = pl.DataFrame({
        'LineId': list(range(1, num_logs + 1)),
        'timestamp': timestamps,
        'node': node_ids,
        'level': levels,
        'component': components,
        'thread': thread_ids,
        'message': logs,
        'e_message_normalized': logs,
    })
    
    # Generate anomaly scores
    anomaly_scores = []
    for i, level in enumerate(levels):
        if level in ["ERROR", "FATAL"]:
            anomaly_scores.append(max(0.8, np.random.uniform(0.75, 0.99)))
        elif level == "WARNING" and np.random.random() < 0.3:
            anomaly_scores.append(np.random.uniform(0.8, 0.95))
        elif np.random.random() < anomaly_ratio:
            anomaly_scores.append(np.random.uniform(0.8, 0.9))
        else:
            anomaly_scores.append(np.random.uniform(0.0, 0.7))
    
    df = df.with_columns(pl.Series(anomaly_scores).alias('anomaly_score'))
    df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias('lexical_context'))
    
    # Print sample anomalies
    print(f"Generated {df.filter(pl.col('anomaly_score') >= 0.8).height} potential anomalies")
    
    return df 