package com.example;

import org.apache.crail.*;
import org.apache.crail.conf.CrailConfiguration;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class CrailKVCacheManager {
    private static final int BUFFER_SIZE = 8 * 1024 * 1024; // 8MB buffer

    // 静态Crail连接实例，用于连接池
    private static CrailStore store = null;
    private static final Object storeLock = new Object();

    // 连接引用计数器
    private static final AtomicInteger connectionRefCount = new AtomicInteger(0);

    // 获取共享的Crail连接
    private static CrailStore getStore() throws Exception {
        synchronized (storeLock) {
            if (store == null) {
                System.err.println("Initializing new Crail connection");

                // 设置配置文件路径
                System.setProperty("crail.conf.dir", "/home/ms-admin/sunshi/crail/conf");
                CrailConfiguration conf = CrailConfiguration.createConfigurationFromFile();
                store = CrailStore.newInstance(conf);

                // 添加JVM关闭钩子来关闭连接
                Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                    try {
                        closeStore();
                    } catch (Exception e) {
                        System.err.println("Failed to close Crail store: " + e.getMessage());
                    }
                }));
            }

            // 增加引用计数
            connectionRefCount.incrementAndGet();
            return store;
        }
    }

    // 关闭共享Crail连接
    private static void closeStore() throws Exception {
        synchronized (storeLock) {
            if (store != null && connectionRefCount.decrementAndGet() <= 0) {
                System.err.println("Closing Crail connection");
                store.close();
                store = null;
                connectionRefCount.set(0);
            }
        }
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            printUsage();
            System.exit(1);
        }

        String operation = args[0].toLowerCase();
        CrailStore storeInstance = null;

        try {
            // 获取共享的Crail连接
            storeInstance = getStore();

            switch (operation) {
                case "upload":
                    if (args.length < 3) {
                        System.err.println("Upload requires <local_file> and <crail_path>");
                        printUsage();
                        System.exit(1);
                    }
                    uploadFile(storeInstance, args[1], args[2]);
                    break;

                case "upload-stream":
                    if (args.length < 2) {
                        System.err.println("Upload-stream requires <crail_path>");
                        printUsage();
                        System.exit(1);
                    }
                    uploadFromStream(storeInstance, args[1]);
                    break;

                case "download":
                    if (args.length < 3) {
                        System.err.println("Download requires <crail_path> and <local_file>");
                        printUsage();
                        System.exit(1);
                    }
                    downloadFile(storeInstance, args[1], args[2]);
                    break;

                case "download-stream":
                    if (args.length < 2) {
                        System.err.println("Download-stream requires <crail_path>");
                        printUsage();
                        System.exit(1);
                    }
                    downloadToStream(storeInstance, args[1]);
                    break;

                case "list":
                    if (args.length < 2) {
                        System.err.println("List requires <crail_directory>");
                        printUsage();
                        System.exit(1);
                    }
                    listDirectory(storeInstance, args[1]);
                    break;

                default:
                    System.err.println("Unknown operation: " + operation);
                    printUsage();
                    System.exit(1);
            }

            // 不关闭连接，由关闭钩子或引用计数控制

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        } finally {
            try {
                // 释放引用
                closeStore();
            } catch (Exception e) {
                System.err.println("Error closing connection: " + e.getMessage());
            }
        }
    }

    private static void printUsage() {
        System.err.println("Usage:");
        System.err.println("  Upload: java -jar crail-kvcache-client.jar upload <local_file> <crail_path>");
        System.err.println("  Upload-Stream: java -jar crail-kvcache-client.jar upload-stream <crail_path>");
        System.err.println("  Download: java -jar crail-kvcache-client.jar download <crail_path> <local_file>");
        System.err.println("  Download-Stream: java -jar crail-kvcache-client.jar download-stream <crail_path>");
        System.err.println("  List: java -jar crail-kvcache-client.jar list <crail_directory>");
    }

    /**
     * 从标准输入流上传数据到Crail
     */
    private static void uploadFromStream(CrailStore store, String crailPath) throws Exception {
        System.err.println("Uploading from stdin to Crail: " + crailPath);

        // 创建Crail文件的父目录（如果需要）
        createParentDirectories(store, crailPath);

        // 创建Crail文件
        CrailNode node = store.create(crailPath, CrailNodeType.DATAFILE,
                CrailStorageClass.DEFAULT, CrailLocationClass.DEFAULT, false).get();
        node.syncDir();

        // 获取Crail文件句柄
        CrailFile crailFile = node.asFile();

        long startTime = System.currentTimeMillis();

        // 预先分配合理的空间大小
        long estimatedSize = 1024 * 1024 * 1024; // 预分配1GB

        // 获取Crail输出流
        CrailBufferedOutputStream outstream = crailFile.getBufferedOutputStream(estimatedSize);

        // 从标准输入读取数据
        ReadableByteChannel inputChannel = Channels.newChannel(System.in);
        ByteBuffer buffer = ByteBuffer.allocateDirect(BUFFER_SIZE);

        long bytesWritten = 0;
        int bytesRead;

        // 读取所有输入数据
        while ((bytesRead = inputChannel.read(buffer)) != -1) {
            buffer.flip();
            bytesWritten += buffer.remaining();
            outstream.write(buffer);
            buffer.clear();

            // 每处理约100MB打印一次进度
            if (bytesWritten % (100 * 1024 * 1024) == 0) {
                System.err.println("Uploaded " + (bytesWritten / (1024 * 1024)) + " MB");
            }
        }

        outstream.close();

        // 注意：Crail API不支持truncate方法，我们依赖输出流正确关闭

        long endTime = System.currentTimeMillis();
        double duration = (endTime - startTime) / 1000.0;
        double speedMBps = (bytesWritten / (1024.0 * 1024.0)) / duration;

        System.err.println("\nUpload completed successfully");
        System.err.printf("Bytes: %d, Time: %.2f sec, Speed: %.2f MB/s\n",
                        bytesWritten, duration, speedMBps);
    }

    /**
     * 将Crail中的数据下载到标准输出流
     */
    private static void downloadToStream(CrailStore store, String crailPath) throws Exception {
        System.err.println("Downloading from Crail to stdout: " + crailPath);

        // 检查Crail文件是否存在
        CrailNode node;
        try {
            node = store.lookup(crailPath).get();
            if (node.getType() != CrailNodeType.DATAFILE) {
                throw new FileNotFoundException("Not a regular file: " + crailPath);
            }
        } catch (Exception e) {
            throw new FileNotFoundException("Crail file not found: " + crailPath);
        }

        // 获取Crail文件
        CrailFile file = node.asFile();
        long fileSize = file.getCapacity();

        System.err.println("File size: " + fileSize + " bytes (" + (fileSize / (1024 * 1024)) + " MB)");

        long startTime = System.currentTimeMillis();

        // 获取Crail输入流
        CrailBufferedInputStream instream = file.getBufferedInputStream(fileSize);

        // 写入到标准输出
        WritableByteChannel outputChannel = Channels.newChannel(System.out);
        ByteBuffer buffer = ByteBuffer.allocateDirect(BUFFER_SIZE);

        long bytesRead = 0;
        int read;

        // 读取所有数据
        while ((read = instream.read(buffer)) > 0) {
            buffer.flip();
            bytesRead += read;
            outputChannel.write(buffer);
            buffer.clear();

            // 每处理约100MB打印一次进度
            if (bytesRead % (100 * 1024 * 1024) == 0) {
                System.err.println("Downloaded " + (bytesRead / (1024 * 1024)) + " MB");
            }
        }

        instream.close();
        System.out.flush();

        long endTime = System.currentTimeMillis();
        double duration = (endTime - startTime) / 1000.0;
        double speedMBps = (bytesRead / (1024.0 * 1024.0)) / duration;

        System.err.println("\nDownload completed successfully");
        System.err.printf("Bytes: %d, Time: %.2f sec, Speed: %.2f MB/s\n",
                         bytesRead, duration, speedMBps);
    }

    private static void uploadFile(CrailStore store, String localPath, String crailPath) throws Exception {
        System.err.println("Uploading file: " + localPath + " -> " + crailPath);

        // 检查本地文件是否存在
        File localFile = new File(localPath);
        if (!localFile.exists() || !localFile.isFile()) {
            throw new FileNotFoundException("Local file not found: " + localPath);
        }

        // 创建Crail文件的父目录（如果需要）
        createParentDirectories(store, crailPath);

        // 创建Crail文件
        CrailNode node = store.create(crailPath, CrailNodeType.DATAFILE,
                CrailStorageClass.DEFAULT, CrailLocationClass.DEFAULT, false).get();
        node.syncDir();

        // 获取Crail文件句柄
        CrailFile crailFile = node.asFile();

        long startTime = System.currentTimeMillis();
        long fileSize = localFile.length();
        long bytesWritten = 0;

        // 获取Crail输出流
        CrailBufferedOutputStream outstream = crailFile.getBufferedOutputStream(fileSize);

        // 使用NIO读取本地文件并写入Crail
        try (FileChannel fileChannel = FileChannel.open(localFile.toPath(), StandardOpenOption.READ)) {
            ByteBuffer buffer = ByteBuffer.allocateDirect(BUFFER_SIZE);

            int bytesRead;
            while ((bytesRead = fileChannel.read(buffer)) != -1) {
                buffer.flip();
                bytesWritten += buffer.remaining();
                outstream.write(buffer);
                buffer.clear();

                // 显示进度
                printProgress("Uploading", bytesWritten, fileSize);
            }
        }

        outstream.close();

        long endTime = System.currentTimeMillis();
        double duration = (endTime - startTime) / 1000.0;
        double speedMBps = (fileSize / (1024.0 * 1024.0)) / duration;

        System.err.println("\nUpload completed successfully");
        System.err.printf("Time: %.2f sec, Speed: %.2f MB/s\n", duration, speedMBps);
    }

    private static void downloadFile(CrailStore store, String crailPath, String localPath) throws Exception {
        System.err.println("Downloading file: " + crailPath + " -> " + localPath);

        // 检查Crail文件是否存在
        CrailNode node;
        try {
            node = store.lookup(crailPath).get();
            if (node.getType() != CrailNodeType.DATAFILE) {
                throw new FileNotFoundException("Not a regular file: " + crailPath);
            }
        } catch (Exception e) {
            throw new FileNotFoundException("Crail file not found: " + crailPath);
        }

        // 获取Crail文件
        CrailFile file = node.asFile();
        long fileSize = file.getCapacity();

        // 创建本地文件的父目录（如果需要）
        Path localFilePath = Paths.get(localPath);
        if (localFilePath.getParent() != null) {
            Files.createDirectories(localFilePath.getParent());
        }

        long startTime = System.currentTimeMillis();
        long bytesRead = 0;

        // 获取Crail输入流
        CrailBufferedInputStream instream = file.getBufferedInputStream(fileSize);

        // 使用NIO写入本地文件
        try (FileChannel fileChannel = FileChannel.open(localFilePath,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {

            ByteBuffer buffer = ByteBuffer.allocateDirect(BUFFER_SIZE);
            int read;

            while ((read = instream.read(buffer)) > 0) {
                buffer.flip();
                bytesRead += read;
                fileChannel.write(buffer);
                buffer.clear();

                printProgress("Downloading", bytesRead, fileSize);
            }
        }

        instream.close();

        long endTime = System.currentTimeMillis();
        double duration = (endTime - startTime) / 1000.0;
        double speedMBps = (fileSize / (1024.0 * 1024.0)) / duration;

        System.err.println("\nDownload completed successfully");
        System.err.printf("Time: %.2f sec, Speed: %.2f MB/s\n", duration, speedMBps);
    }

    private static void listDirectory(CrailStore store, String crailDirPath) throws Exception {
        System.err.println("Listing directory: " + crailDirPath);

        if (!crailDirPath.endsWith("/")) {
            crailDirPath += "/";
        }

        try {
            CrailNode dirNode = store.lookup(crailDirPath).get();
            if (dirNode.getType() != CrailNodeType.DIRECTORY) {
                System.err.println("Not a directory: " + crailDirPath);
                return;
            }

            CrailDirectory dir = dirNode.asDirectory();
            Iterator<String> entriesIterator = dir.listEntries();

            List<String> entries = new ArrayList<>();
            while (entriesIterator.hasNext()) {
                entries.add(entriesIterator.next());
            }

            System.err.println("Found " + entries.size() + " entries:");
            for (String entry : entries) {
                CrailNode node = store.lookup(crailDirPath + entry).get();
                String type = (node.getType() == CrailNodeType.DIRECTORY) ? "DIR" : "FILE";
                long size = (node.getType() == CrailNodeType.DATAFILE) ? node.asFile().getCapacity() : 0;

                System.err.printf("%-8s %-12d %s\n", type, size, entry);
            }

        } catch (Exception e) {
            System.err.println("Directory does not exist: " + crailDirPath);
            e.printStackTrace();
        }
    }

    private static void createParentDirectories(CrailStore store, String path) throws Exception {
        String[] parts = path.split("/");
        StringBuilder currentPath = new StringBuilder();

        for (int i = 0; i < parts.length - 1; i++) {
            if (parts[i].isEmpty()) continue;

            currentPath.append("/").append(parts[i]);
            String dirPath = currentPath.toString();

            try {
                CrailNode node = store.lookup(dirPath).get();
                if (node.getType() != CrailNodeType.DIRECTORY) {
                    throw new IOException("Path exists but is not a directory: " + dirPath);
                }
            } catch (Exception e) {
                try {
                    store.create(dirPath, CrailNodeType.DIRECTORY,
                            CrailStorageClass.DEFAULT, CrailLocationClass.DEFAULT, false).get();
                    System.err.println("Created directory: " + dirPath);
                } catch (Exception ex) {
                    // 检查目录是否已经存在（可能由另一个进程创建）
                    CrailNode node = store.lookup(dirPath).get();
                    if (node.getType() != CrailNodeType.DIRECTORY) {
                        throw new IOException("Failed to create directory: " + dirPath);
                    }
                }
            }
        }
    }

    private static void printProgress(String action, long current, long total) {
        int percent = (int) (100.0 * current / total);
        int progressChars = percent / 2;

        StringBuilder progressBar = new StringBuilder("[");
        for (int i = 0; i < 50; i++) {
            if (i < progressChars) {
                progressBar.append("=");
            } else if (i == progressChars) {
                progressBar.append(">");
            } else {
                progressBar.append(" ");
            }
        }
        progressBar.append("]");

        String formattedProgress = String.format("%s %s %d%% (%d/%d bytes)",
                action, progressBar.toString(), percent, current, total);

        System.err.print("\r" + formattedProgress);
        System.err.flush();
    }
}
