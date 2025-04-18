"""
System Metrics Exporter for Prometheus

This module collects and exports system-level metrics including CPU, memory,
disk usage, and network statistics to Prometheus.
"""

import logging
import os
import time
import platform
import psutil
from typing import Dict, Any, List, Optional, Union

from .base_exporter import BaseExporter

logger = logging.getLogger(__name__)

class SystemMetricsExporter(BaseExporter):
    """
    Exporter for system resource metrics.
    Collects information about CPU, memory, disk, and network usage.
    """
    
    def __init__(self, port: int = 8001, interval: int = 15):
        """
        Initialize the system metrics exporter.
        
        Args:
            port: Port to expose metrics on
            interval: Collection interval in seconds
        """
        super().__init__(name="system", port=port, interval=interval)
        
        # Initialize metrics
        self._init_cpu_metrics()
        self._init_memory_metrics()
        self._init_disk_metrics()
        self._init_network_metrics()
        self._init_process_metrics()
        
        logger.info("Initialized system metrics exporter")
    
    def _init_cpu_metrics(self) -> None:
        """Initialize CPU metrics."""
        self.create_gauge(
            name="cpu_usage_percent", 
            description="CPU usage percentage (overall)",
        )
        
        self.create_gauge(
            name="cpu_usage_per_core_percent",
            description="CPU usage percentage per core",
            labels=["core"]
        )
        
        self.create_gauge(
            name="cpu_load_1m",
            description="CPU load average over 1 minute"
        )
        
        self.create_gauge(
            name="cpu_load_5m",
            description="CPU load average over 5 minutes"
        )
        
        self.create_gauge(
            name="cpu_load_15m",
            description="CPU load average over 15 minutes"
        )
    
    def _init_memory_metrics(self) -> None:
        """Initialize memory metrics."""
        self.create_gauge(
            name="memory_total_bytes",
            description="Total system memory in bytes"
        )
        
        self.create_gauge(
            name="memory_available_bytes",
            description="Available system memory in bytes"
        )
        
        self.create_gauge(
            name="memory_used_bytes",
            description="Used system memory in bytes"
        )
        
        self.create_gauge(
            name="memory_used_percent",
            description="Used system memory percentage"
        )
        
        self.create_gauge(
            name="swap_total_bytes",
            description="Total swap memory in bytes"
        )
        
        self.create_gauge(
            name="swap_used_bytes",
            description="Used swap memory in bytes"
        )
        
        self.create_gauge(
            name="swap_used_percent",
            description="Used swap memory percentage"
        )
    
    def _init_disk_metrics(self) -> None:
        """Initialize disk metrics."""
        self.create_gauge(
            name="disk_total_bytes",
            description="Total disk space in bytes",
            labels=["device", "mountpoint"]
        )
        
        self.create_gauge(
            name="disk_used_bytes",
            description="Used disk space in bytes",
            labels=["device", "mountpoint"]
        )
        
        self.create_gauge(
            name="disk_used_percent",
            description="Used disk space percentage",
            labels=["device", "mountpoint"]
        )
        
        self.create_gauge(
            name="disk_io_read_bytes",
            description="Total disk read bytes",
            labels=["device"]
        )
        
        self.create_gauge(
            name="disk_io_write_bytes",
            description="Total disk write bytes",
            labels=["device"]
        )
        
        self.create_gauge(
            name="disk_io_read_count",
            description="Total disk read operations",
            labels=["device"]
        )
        
        self.create_gauge(
            name="disk_io_write_count",
            description="Total disk write operations",
            labels=["device"]
        )
    
    def _init_network_metrics(self) -> None:
        """Initialize network metrics."""
        self.create_gauge(
            name="network_bytes_sent",
            description="Total network bytes sent",
            labels=["interface"]
        )
        
        self.create_gauge(
            name="network_bytes_recv",
            description="Total network bytes received",
            labels=["interface"]
        )
        
        self.create_gauge(
            name="network_packets_sent",
            description="Total network packets sent",
            labels=["interface"]
        )
        
        self.create_gauge(
            name="network_packets_recv",
            description="Total network packets received",
            labels=["interface"]
        )
        
        self.create_gauge(
            name="network_errin",
            description="Total network errors on receiving",
            labels=["interface"]
        )
        
        self.create_gauge(
            name="network_errout",
            description="Total network errors on sending",
            labels=["interface"]
        )
    
    def _init_process_metrics(self) -> None:
        """Initialize process metrics for the current process."""
        self.create_gauge(
            name="process_cpu_percent",
            description="CPU usage percentage of the trading system process"
        )
        
        self.create_gauge(
            name="process_memory_rss",
            description="Resident set size of the trading system process in bytes"
        )
        
        self.create_gauge(
            name="process_memory_vms",
            description="Virtual memory size of the trading system process in bytes"
        )
        
        self.create_gauge(
            name="process_threads",
            description="Number of threads in the trading system process"
        )
        
        self.create_gauge(
            name="process_open_files",
            description="Number of open files by the trading system process"
        )
    
    def collect(self) -> None:
        """Collect and update all system metrics."""
        try:
            self._collect_cpu_metrics()
            self._collect_memory_metrics()
            self._collect_disk_metrics()
            self._collect_network_metrics()
            self._collect_process_metrics()
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    def _collect_cpu_metrics(self) -> None:
        """Collect CPU metrics."""
        try:
            # Overall CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.get_metric("cpu_usage_percent").set(cpu_percent)
            
            # Per-core CPU usage
            per_cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
            cpu_gauge = self.get_metric("cpu_usage_per_core_percent")
            for i, cpu_percent in enumerate(per_cpu_percent):
                cpu_gauge.labels(core=str(i)).set(cpu_percent)
            
            # CPU load averages
            if platform.system() != "Windows":  # Load averages not available on Windows
                load1, load5, load15 = os.getloadavg()
                self.get_metric("cpu_load_1m").set(load1)
                self.get_metric("cpu_load_5m").set(load5)
                self.get_metric("cpu_load_15m").set(load15)
                
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {str(e)}")
    
    def _collect_memory_metrics(self) -> None:
        """Collect memory metrics."""
        try:
            # Virtual memory
            virtual_memory = psutil.virtual_memory()
            self.get_metric("memory_total_bytes").set(virtual_memory.total)
            self.get_metric("memory_available_bytes").set(virtual_memory.available)
            self.get_metric("memory_used_bytes").set(virtual_memory.used)
            self.get_metric("memory_used_percent").set(virtual_memory.percent)
            
            # Swap memory
            swap_memory = psutil.swap_memory()
            self.get_metric("swap_total_bytes").set(swap_memory.total)
            self.get_metric("swap_used_bytes").set(swap_memory.used)
            self.get_metric("swap_used_percent").set(swap_memory.percent)
            
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {str(e)}")
    
    def _collect_disk_metrics(self) -> None:
        """Collect disk metrics."""
        try:
            # Disk usage
            disk_partitions = psutil.disk_partitions()
            disk_total_gauge = self.get_metric("disk_total_bytes")
            disk_used_gauge = self.get_metric("disk_used_bytes")
            disk_percent_gauge = self.get_metric("disk_used_percent")
            
            for partition in disk_partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    
                    disk_total_gauge.labels(
                        device=partition.device,
                        mountpoint=partition.mountpoint
                    ).set(usage.total)
                    
                    disk_used_gauge.labels(
                        device=partition.device,
                        mountpoint=partition.mountpoint
                    ).set(usage.used)
                    
                    disk_percent_gauge.labels(
                        device=partition.device,
                        mountpoint=partition.mountpoint
                    ).set(usage.percent)
                    
                except PermissionError:
                    # Skip partitions that can't be accessed
                    continue
            
            # Disk I/O counters
            disk_io = psutil.disk_io_counters(perdisk=True)
            disk_read_bytes = self.get_metric("disk_io_read_bytes")
            disk_write_bytes = self.get_metric("disk_io_write_bytes")
            disk_read_count = self.get_metric("disk_io_read_count")
            disk_write_count = self.get_metric("disk_io_write_count")
            
            for disk_name, counters in disk_io.items():
                disk_read_bytes.labels(device=disk_name).set(counters.read_bytes)
                disk_write_bytes.labels(device=disk_name).set(counters.write_bytes)
                disk_read_count.labels(device=disk_name).set(counters.read_count)
                disk_write_count.labels(device=disk_name).set(counters.write_count)
                
        except Exception as e:
            logger.error(f"Error collecting disk metrics: {str(e)}")
    
    def _collect_network_metrics(self) -> None:
        """Collect network metrics."""
        try:
            network_io = psutil.net_io_counters(pernic=True)
            bytes_sent = self.get_metric("network_bytes_sent")
            bytes_recv = self.get_metric("network_bytes_recv")
            packets_sent = self.get_metric("network_packets_sent")
            packets_recv = self.get_metric("network_packets_recv")
            errin = self.get_metric("network_errin")
            errout = self.get_metric("network_errout")
            
            for interface, counters in network_io.items():
                bytes_sent.labels(interface=interface).set(counters.bytes_sent)
                bytes_recv.labels(interface=interface).set(counters.bytes_recv)
                packets_sent.labels(interface=interface).set(counters.packets_sent)
                packets_recv.labels(interface=interface).set(counters.packets_recv)
                errin.labels(interface=interface).set(counters.errin)
                errout.labels(interface=interface).set(counters.errout)
                
        except Exception as e:
            logger.error(f"Error collecting network metrics: {str(e)}")
    
    def _collect_process_metrics(self) -> None:
        """Collect metrics for the current process."""
        try:
            process = psutil.Process(os.getpid())
            
            # CPU usage
            self.get_metric("process_cpu_percent").set(process.cpu_percent())
            
            # Memory usage
            memory_info = process.memory_info()
            self.get_metric("process_memory_rss").set(memory_info.rss)
            self.get_metric("process_memory_vms").set(memory_info.vms)
            
            # Thread count
            self.get_metric("process_threads").set(process.num_threads())
            
            # Open files
            try:
                open_files = len(process.open_files())
                self.get_metric("process_open_files").set(open_files)
            except psutil.AccessDenied:
                # Skip if no permission to access open files
                pass
                
        except Exception as e:
            logger.error(f"Error collecting process metrics: {str(e)}")
