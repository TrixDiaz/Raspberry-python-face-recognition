# React Native Integration Guide

This guide shows how to integrate the Raspberry Pi camera stream into your React Native mobile app.

## 🚀 Quick Start

### 1. Start the Standalone Stream

```bash
python run_standalone_safe.py
```

### 2. Get Network Information

```bash
curl http://[YOUR_PI_IP]:8080/api/network
```

This will return the server IP and available stream URLs.

## 📱 React Native Implementation

### Installation

```bash
npm install react-native-video
# or
yarn add react-native-video
```

### Basic Video Component

```jsx
import React, {useState, useEffect} from "react";
import {View, Text, StyleSheet, Alert} from "react-native";
import Video from "react-native-video";

const CameraStream = () => {
  const [streamUrl, setStreamUrl] = useState("");
  const [serverInfo, setServerInfo] = useState(null);

  useEffect(() => {
    // Get server information
    fetchServerInfo();
  }, []);

  const fetchServerInfo = async () => {
    try {
      // Replace with your Pi's IP address
      const response = await fetch("http://192.168.1.100:8080/api/network");
      const data = await response.json();
      setServerInfo(data);
      setStreamUrl(data.stream_urls.mp4); // Use MP4 stream for React Native
    } catch (error) {
      console.error("Failed to fetch server info:", error);
      Alert.alert("Connection Error", "Cannot connect to camera server");
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Raspberry Pi Camera</Text>

      {streamUrl ? (
        <Video
          source={{uri: streamUrl}}
          style={styles.video}
          controls={true}
          resizeMode="contain"
          onError={(error) => {
            console.error("Video error:", error);
            Alert.alert("Stream Error", "Cannot load camera stream");
          }}
          onLoad={() => console.log("Video loaded successfully")}
        />
      ) : (
        <Text style={styles.loading}>Loading camera stream...</Text>
      )}

      {serverInfo && (
        <View style={styles.info}>
          <Text style={styles.infoText}>Server: {serverInfo.server_ip}</Text>
          <Text style={styles.infoText}>Port: {serverInfo.port}</Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
    justifyContent: "center",
    alignItems: "center",
  },
  title: {
    color: "#fff",
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 20,
  },
  video: {
    width: "100%",
    height: 300,
  },
  loading: {
    color: "#fff",
    fontSize: 16,
  },
  info: {
    marginTop: 20,
  },
  infoText: {
    color: "#fff",
    fontSize: 14,
  },
});

export default CameraStream;
```

### Advanced Implementation with Controls

```jsx
import React, {useState, useEffect} from "react";
import {View, Text, StyleSheet, TouchableOpacity, Alert} from "react-native";
import Video from "react-native-video";

const AdvancedCameraStream = () => {
  const [streamUrl, setStreamUrl] = useState("");
  const [isPlaying, setIsPlaying] = useState(true);
  const [streamType, setStreamType] = useState("mp4");
  const [serverInfo, setServerInfo] = useState(null);

  useEffect(() => {
    fetchServerInfo();
  }, []);

  const fetchServerInfo = async () => {
    try {
      const response = await fetch("http://192.168.1.100:8080/api/network");
      const data = await response.json();
      setServerInfo(data);
      setStreamUrl(data.stream_urls[streamType]);
    } catch (error) {
      console.error("Failed to fetch server info:", error);
      Alert.alert("Connection Error", "Cannot connect to camera server");
    }
  };

  const switchStreamType = (type) => {
    setStreamType(type);
    if (serverInfo) {
      setStreamUrl(serverInfo.stream_urls[type]);
    }
  };

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Raspberry Pi Camera Stream</Text>

      {streamUrl ? (
        <Video
          source={{uri: streamUrl}}
          style={styles.video}
          controls={true}
          resizeMode="contain"
          paused={!isPlaying}
          onError={(error) => {
            console.error("Video error:", error);
            Alert.alert("Stream Error", "Cannot load camera stream");
          }}
          onLoad={() => console.log("Video loaded successfully")}
        />
      ) : (
        <Text style={styles.loading}>Loading camera stream...</Text>
      )}

      <View style={styles.controls}>
        <TouchableOpacity
          style={[
            styles.button,
            isPlaying ? styles.pauseButton : styles.playButton,
          ]}
          onPress={togglePlayPause}
        >
          <Text style={styles.buttonText}>{isPlaying ? "Pause" : "Play"}</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.button,
            streamType === "mp4" ? styles.activeButton : styles.inactiveButton,
          ]}
          onPress={() => switchStreamType("mp4")}
        >
          <Text style={styles.buttonText}>MP4</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.button,
            streamType === "h264" ? styles.activeButton : styles.inactiveButton,
          ]}
          onPress={() => switchStreamType("h264")}
        >
          <Text style={styles.buttonText}>H.264</Text>
        </TouchableOpacity>
      </View>

      {serverInfo && (
        <View style={styles.info}>
          <Text style={styles.infoText}>Server: {serverInfo.server_ip}</Text>
          <Text style={styles.infoText}>
            Stream Type: {streamType.toUpperCase()}
          </Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#000",
    justifyContent: "center",
    alignItems: "center",
  },
  title: {
    color: "#fff",
    fontSize: 24,
    fontWeight: "bold",
    marginBottom: 20,
  },
  video: {
    width: "100%",
    height: 300,
  },
  loading: {
    color: "#fff",
    fontSize: 16,
  },
  controls: {
    flexDirection: "row",
    marginTop: 20,
    gap: 10,
  },
  button: {
    padding: 10,
    borderRadius: 5,
    minWidth: 60,
    alignItems: "center",
  },
  playButton: {
    backgroundColor: "#4CAF50",
  },
  pauseButton: {
    backgroundColor: "#f44336",
  },
  activeButton: {
    backgroundColor: "#2196F3",
  },
  inactiveButton: {
    backgroundColor: "#666",
  },
  buttonText: {
    color: "#fff",
    fontWeight: "bold",
  },
  info: {
    marginTop: 20,
  },
  infoText: {
    color: "#fff",
    fontSize: 14,
  },
});

export default AdvancedCameraStream;
```

## 🔧 Android Studio Integration

### For Android Studio projects:

```java
// In your Android activity
private void setupVideoView() {
    VideoView videoView = findViewById(R.id.videoView);

    // Use RTSP stream for Android Studio
    String streamUrl = "http://192.168.1.100:8080/stream/rtsp";

    videoView.setVideoURI(Uri.parse(streamUrl));
    videoView.setMediaController(new MediaController(this));
    videoView.start();
}
```

### XML Layout:

```xml
<VideoView
    android:id="@+id/videoView"
    android:layout_width="match_parent"
    android:layout_height="300dp"
    android:layout_centerInParent="true" />
```

## 🌐 Network Configuration

### Finding Your Pi's IP Address

```bash
# On the Raspberry Pi
hostname -I
```

### Testing Connection

```bash
# Test from your mobile device/computer
curl http://[PI_IP]:8080/api/network
```

### Firewall Configuration

```bash
# Allow port 8080 through firewall
sudo ufw allow 8080
```

## 📱 Mobile App Permissions

### Android (android/app/src/main/AndroidManifest.xml)

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
```

### iOS (ios/YourApp/Info.plist)

```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>
```

## 🔍 Troubleshooting

### Common Issues

1. **Connection Refused**

   - Check if Pi and mobile device are on same network
   - Verify Pi's IP address
   - Check firewall settings

2. **Video Not Loading**

   - Try different stream types (mp4, h264, rtsp)
   - Check network connectivity
   - Verify server is running

3. **Poor Performance**
   - Use MP4 stream for React Native
   - Use RTSP stream for Android Studio
   - Check network bandwidth

### Debug Commands

```bash
# Check server status
curl http://[PI_IP]:8080/status

# Test stream endpoints
curl -I http://[PI_IP]:8080/stream/mp4
curl -I http://[PI_IP]:8080/stream/h264
```

## 🚀 Production Deployment

### For Production Use:

1. **Security**: Add authentication to stream endpoints
2. **Performance**: Use H.264 encoding for better compression
3. **Reliability**: Implement reconnection logic
4. **Monitoring**: Add error handling and logging

### Example Production Code:

```jsx
const ProductionCameraStream = () => {
  const [streamUrl, setStreamUrl] = useState("");
  const [retryCount, setRetryCount] = useState(0);
  const [isConnected, setIsConnected] = useState(false);

  const connectToStream = async () => {
    try {
      const response = await fetch("http://192.168.1.100:8080/api/network");
      const data = await response.json();
      setStreamUrl(data.stream_urls.mp4);
      setIsConnected(true);
      setRetryCount(0);
    } catch (error) {
      console.error("Connection failed:", error);
      if (retryCount < 3) {
        setTimeout(() => {
          setRetryCount(retryCount + 1);
          connectToStream();
        }, 2000);
      }
    }
  };

  useEffect(() => {
    connectToStream();
  }, []);

  // ... rest of component
};
```

Your React Native app should now be able to connect to the Raspberry Pi camera stream! 🎥📱
