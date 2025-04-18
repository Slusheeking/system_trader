#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Redis Metrics Exporter
---------------------
Exports Redis metrics to Prometheus.
"""

import time
import logging
import redis
from prometheus_client import start_http_server, Gauge, Counter, Info

from monitoring.prometheus.exporters.base_exporter import BaseExporter
from data.database.redis_client import get_redis_client

# Setup logging
logger = logging.getLogger(__name__)

class RedisMetricsExporter(BaseExporter):
    """
    Exporter for Redis metrics.
    """

    def __init__(self, config=None):
        """
        Initialize the Redis metrics exporter.

        Args:
            config: Configuration dictionary
        """
        super().__init__(name="redis_metrics", config=config)

        # Default configuration
        self.redis_host = self.config.get('redis_host', 'localhost')
        self.redis_port = self.config.get('redis_port', 6379)
        self.redis_db = self.config.get('redis_db', 0)
        self.redis_password = self.config.get('redis_password', None)
        self.exporter_port = self.config.get('exporter_port', 9121)
        self.scrape_interval = self.config.get('scrape_interval', 15)

        # Redis client
        self.redis_client = get_redis_client()

        # Prometheus metrics
        self.metrics = self._create_metrics()

        logger.info(f"Redis metrics exporter initialized for {self.redis_host}:{self.redis_port}")

    def _create_metrics(self):
        """
        Create Prometheus metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Redis info metrics
        metrics['info'] = Info('redis_info', 'Redis server information')

        # Memory metrics
        metrics['memory_used_bytes'] = Gauge('redis_memory_used_bytes', 'Redis memory used in bytes')
        metrics['memory_peak_bytes'] = Gauge('redis_memory_peak_bytes', 'Redis memory peak in bytes')
        metrics['memory_fragmentation_ratio'] = Gauge('redis_memory_fragmentation_ratio', 'Redis memory fragmentation ratio')
        metrics['memory_rss_bytes'] = Gauge('redis_memory_rss_bytes', 'Redis memory RSS in bytes')

        # Client metrics
        metrics['connected_clients'] = Gauge('redis_connected_clients', 'Redis connected clients')
        metrics['blocked_clients'] = Gauge('redis_blocked_clients', 'Redis blocked clients')

        # Command metrics
        metrics['commands_processed'] = Counter('redis_commands_processed_total', 'Redis commands processed')
        metrics['commands_per_second'] = Gauge('redis_commands_per_second', 'Redis commands per second')

        # Key metrics
        metrics['keys_total'] = Gauge('redis_keys_total', 'Redis total keys', ['db'])
        metrics['keys_expired'] = Counter('redis_keys_expired_total', 'Redis expired keys')
        metrics['keys_evicted'] = Counter('redis_keys_evicted_total', 'Redis evicted keys')

        # Connection metrics
        metrics['connections_received'] = Counter('redis_connections_received_total', 'Redis connections received')
        metrics['connections_rejected'] = Counter('redis_connections_rejected_total', 'Redis connections rejected')

        # Network metrics
        metrics['network_input_bytes'] = Counter('redis_network_input_bytes_total', 'Redis network input bytes')
        metrics['network_output_bytes'] = Counter('redis_network_output_bytes_total', 'Redis network output bytes')

        # Persistence metrics
        metrics['rdb_changes_since_last_save'] = Gauge('redis_rdb_changes_since_last_save', 'Redis RDB changes since last save')
        metrics['rdb_last_save_time'] = Gauge('redis_rdb_last_save_time', 'Redis RDB last save time')
        metrics['rdb_last_bgsave_status'] = Gauge('redis_rdb_last_bgsave_status', 'Redis RDB last background save status (1 = success, 0 = failure)')
        metrics['aof_last_rewrite_status'] = Gauge('redis_aof_last_rewrite_status', 'Redis AOF last rewrite status (1 = success, 0 = failure)')

        # Replication metrics
        metrics['connected_slaves'] = Gauge('redis_connected_slaves', 'Redis connected slaves')
        metrics['replication_backlog_size'] = Gauge('redis_replication_backlog_size', 'Redis replication backlog size')

        # Latency metrics
        metrics['latency_ms'] = Gauge('redis_latency_ms', 'Redis latency in milliseconds')

        # Cache metrics
        metrics['cache_hit_ratio'] = Gauge('redis_cache_hit_ratio', 'Redis cache hit ratio')
        metrics['cache_hits'] = Counter('redis_cache_hits_total', 'Redis cache hits')
        metrics['cache_misses'] = Counter('redis_cache_misses_total', 'Redis cache misses')

        # System metrics
        metrics['uptime_seconds'] = Gauge('redis_uptime_seconds', 'Redis uptime in seconds')
        metrics['cpu_sys_seconds'] = Counter('redis_cpu_sys_seconds_total', 'Redis CPU system seconds')
        metrics['cpu_user_seconds'] = Counter('redis_cpu_user_seconds_total', 'Redis CPU user seconds')

        return metrics

    def _collect_metrics(self):
        """
        Collect Redis metrics.
        """
        try:
            # Get Redis info
            info = self.redis_client.redis.info()

            # Update Redis info metric
            self.metrics['info'].info({
                'redis_version': info.get('redis_version', 'unknown'),
                'redis_mode': info.get('redis_mode', 'unknown'),
                'os': info.get('os', 'unknown'),
                'arch_bits': str(info.get('arch_bits', 'unknown')),
                'multiplexing_api': info.get('multiplexing_api', 'unknown'),
                'gcc_version': info.get('gcc_version', 'unknown'),
                'process_id': str(info.get('process_id', 'unknown')),
                'run_id': info.get('run_id', 'unknown'),
                'tcp_port': str(info.get('tcp_port', 'unknown')),
                'uptime_in_seconds': str(info.get('uptime_in_seconds', 'unknown')),
                'uptime_in_days': str(info.get('uptime_in_days', 'unknown')),
                'hz': str(info.get('hz', 'unknown')),
                'configured_hz': str(info.get('configured_hz', 'unknown')),
                'lru_clock': str(info.get('lru_clock', 'unknown')),
                'executable': info.get('executable', 'unknown'),
                'config_file': info.get('config_file', 'unknown'),
            })

            # Update memory metrics
            self.metrics['memory_used_bytes'].set(info.get('used_memory', 0))
            self.metrics['memory_peak_bytes'].set(info.get('used_memory_peak', 0))
            self.metrics['memory_fragmentation_ratio'].set(info.get('mem_fragmentation_ratio', 0))
            self.metrics['memory_rss_bytes'].set(info.get('used_memory_rss', 0))

            # Update client metrics
            self.metrics['connected_clients'].set(info.get('connected_clients', 0))
            self.metrics['blocked_clients'].set(info.get('blocked_clients', 0))

            # Update command metrics
            self.metrics['commands_processed']._value.set(info.get('total_commands_processed', 0))
            self.metrics['commands_per_second'].set(info.get('instantaneous_ops_per_sec', 0))

            # Update key metrics
            for db in info:
                if db.startswith('db'):
                    db_info = info[db]
                    self.metrics['keys_total'].labels(db=db).set(db_info.get('keys', 0))
            self.metrics['keys_expired']._value.set(info.get('expired_keys', 0))
            self.metrics['keys_evicted']._value.set(info.get('evicted_keys', 0))

            # Update connection metrics
            self.metrics['connections_received']._value.set(info.get('total_connections_received', 0))
            self.metrics['connections_rejected']._value.set(info.get('rejected_connections', 0))

            # Update network metrics
            self.metrics['network_input_bytes']._value.set(info.get('total_net_input_bytes', 0))
            self.metrics['network_output_bytes']._value.set(info.get('total_net_output_bytes', 0))

            # Update persistence metrics
            self.metrics['rdb_changes_since_last_save'].set(info.get('rdb_changes_since_last_save', 0))
            self.metrics['rdb_last_save_time'].set(info.get('rdb_last_save_time', 0))
            self.metrics['rdb_last_bgsave_status'].set(1 if info.get('rdb_last_bgsave_status', '') == 'ok' else 0)
            self.metrics['aof_last_rewrite_status'].set(1 if info.get('aof_last_rewrite_status', '') == 'ok' else 0)

            # Update replication metrics
            self.metrics['connected_slaves'].set(info.get('connected_slaves', 0))
            self.metrics['replication_backlog_size'].set(info.get('repl_backlog_size', 0))

            # Update latency metrics
            start_time = time.time()
            self.redis_client.redis.ping()
            latency = (time.time() - start_time) * 1000
            self.metrics['latency_ms'].set(latency)

            # Update cache metrics
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            total = hits + misses
            hit_ratio = hits / total if total > 0 else 0
            self.metrics['cache_hit_ratio'].set(hit_ratio)
            self.metrics['cache_hits']._value.set(hits)
            self.metrics['cache_misses']._value.set(misses)

            # Update system metrics
            self.metrics['uptime_seconds'].set(info.get('uptime_in_seconds', 0))
            self.metrics['cpu_sys_seconds']._value.set(info.get('used_cpu_sys', 0))
            self.metrics['cpu_user_seconds']._value.set(info.get('used_cpu_user', 0))

            logger.debug("Redis metrics collected successfully")
        except Exception as e:
            logger.error(f"Error collecting Redis metrics: {str(e)}")

    def start(self):
        """
        Start the Redis metrics exporter.
        """
        try:
            # Start HTTP server for Prometheus
            start_http_server(self.exporter_port)
            logger.info(f"Redis metrics exporter started on port {self.exporter_port}")

            # Collect metrics periodically
            while True:
                self._collect_metrics()
                time.sleep(self.scrape_interval)
        except Exception as e:
            logger.error(f"Error starting Redis metrics exporter: {str(e)}")
            raise

    def stop(self):
        """
        Stop the Redis metrics exporter.
        """
        logger.info("Redis metrics exporter stopped")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Redis Metrics Exporter')
    parser.add_argument('--host', type=str, default='localhost', help='Redis host')
    parser.add_argument('--port', type=int, default=6379, help='Redis port')
    parser.add_argument('--db', type=int, default=0, help='Redis database')
    parser.add_argument('--password', type=str, default=None, help='Redis password')
    parser.add_argument('--exporter-port', type=int, default=9121, help='Exporter port')
    parser.add_argument('--scrape-interval', type=int, default=15, help='Scrape interval in seconds')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create config from args
    config = {
        'redis_host': args.host,
        'redis_port': args.port,
        'redis_db': args.db,
        'redis_password': args.password,
        'exporter_port': args.exporter_port,
        'scrape_interval': args.scrape_interval
    }

    # Create and start exporter
    exporter = RedisMetricsExporter(config)
    exporter.start()
