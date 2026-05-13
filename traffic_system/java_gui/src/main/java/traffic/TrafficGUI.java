package traffic;

import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.*;
import javax.imageio.ImageIO;
import org.json.JSONObject;

public class TrafficGUI extends JFrame {
    private static final String BACKEND = "http://localhost:5000";
    private JLabel videoLabel, totalLabel, carLabel, bikeLabel, busLabel, truckLabel;
    private JLabel densityLabel, modeLabel, elapsedLabel;
    private JPanel densityPanel;
    private JTextArea logArea;
    private volatile boolean running = true;

    public TrafficGUI() {
        super("AI Traffic Vehicle Counter");
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setSize(1280, 800);
        setLocationRelativeTo(null);
        try { UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName()); } catch(Exception e){}
        buildUI();
        startVideoThread();
        startStatsThread();
    }

    private void buildUI() {
        JPanel main = new JPanel(new BorderLayout(8,8));
        main.setBackground(new Color(30,30,40));
        main.setBorder(BorderFactory.createEmptyBorder(10,10,10,10));

        // Header
        JPanel hdr = new JPanel(new BorderLayout());
        hdr.setBackground(new Color(25,25,35));
        hdr.setBorder(BorderFactory.createMatteBorder(0,0,2,0,new Color(0,150,255)));
        JLabel title = new JLabel("  AI Traffic Vehicle Counter & Classifier");
        title.setFont(new Font("Segoe UI",Font.BOLD,18));
        title.setForeground(new Color(0,200,255));
        hdr.add(title, BorderLayout.WEST);
        modeLabel = new JLabel("Mode: SVM  ");
        modeLabel.setForeground(new Color(180,180,200));
        elapsedLabel = new JLabel("0s  ");
        elapsedLabel.setForeground(new Color(180,180,200));
        JPanel rh = new JPanel(new FlowLayout(FlowLayout.RIGHT));
        rh.setOpaque(false); rh.add(modeLabel); rh.add(elapsedLabel);
        hdr.add(rh, BorderLayout.EAST);
        main.add(hdr, BorderLayout.NORTH);

        // Video
        videoLabel = new JLabel("Connecting...", SwingConstants.CENTER);
        videoLabel.setForeground(Color.GRAY);
        videoLabel.setBackground(Color.BLACK);
        videoLabel.setOpaque(true);
        videoLabel.setPreferredSize(new Dimension(800,600));
        main.add(videoLabel, BorderLayout.CENTER);

        // Right panel
        JPanel rp = new JPanel();
        rp.setLayout(new BoxLayout(rp, BoxLayout.Y_AXIS));
        rp.setBackground(new Color(35,35,50));
        rp.setBorder(BorderFactory.createEmptyBorder(15,15,15,15));
        rp.setPreferredSize(new Dimension(260,0));

        rp.add(label("LIVE STATS", new Color(0,200,255), 16, true));
        rp.add(Box.createVerticalStrut(10));
        totalLabel = statLabel("Total: 0", new Color(255,220,0));
        carLabel = statLabel("Cars: 0", new Color(0,200,100));
        bikeLabel = statLabel("Bikes: 0", new Color(255,100,100));
        busLabel = statLabel("Buses: 0", new Color(100,150,255));
        truckLabel = statLabel("Trucks: 0", new Color(255,160,50));
        rp.add(totalLabel); rp.add(Box.createVerticalStrut(4));
        rp.add(carLabel); rp.add(Box.createVerticalStrut(4));
        rp.add(bikeLabel); rp.add(Box.createVerticalStrut(4));
        rp.add(busLabel); rp.add(Box.createVerticalStrut(4));
        rp.add(truckLabel); rp.add(Box.createVerticalStrut(15));

        rp.add(label("DENSITY", new Color(200,200,220), 14, true));
        rp.add(Box.createVerticalStrut(5));
        densityPanel = new JPanel(new BorderLayout());
        densityPanel.setMaximumSize(new Dimension(240,40));
        densityPanel.setBackground(new Color(0,180,0));
        densityLabel = new JLabel("LOW", SwingConstants.CENTER);
        densityLabel.setFont(new Font("Segoe UI",Font.BOLD,16));
        densityLabel.setForeground(Color.WHITE);
        densityPanel.add(densityLabel);
        densityPanel.setAlignmentX(LEFT_ALIGNMENT);
        rp.add(densityPanel); rp.add(Box.createVerticalStrut(20));

        rp.add(label("CONTROLS", new Color(200,200,220), 14, true));
        rp.add(Box.createVerticalStrut(8));
        rp.add(btn("SVM Mode", new Color(0,120,200), e->setMode("svm")));
        rp.add(Box.createVerticalStrut(4));
        rp.add(btn("YOLO Mode", new Color(200,80,0), e->setMode("yolo")));
        rp.add(Box.createVerticalStrut(4));
        rp.add(btn("Upload Video", new Color(80,80,150), e->upload()));
        rp.add(Box.createVerticalStrut(4));
        rp.add(btn("Reset", new Color(180,30,30), e->reset()));
        rp.add(Box.createVerticalGlue());
        main.add(rp, BorderLayout.EAST);

        // Bottom log
        JPanel bp = new JPanel(new BorderLayout());
        bp.setBackground(new Color(25,25,35));
        bp.setPreferredSize(new Dimension(0,90));
        logArea = new JTextArea();
        logArea.setEditable(false);
        logArea.setBackground(new Color(20,20,30));
        logArea.setForeground(new Color(0,255,100));
        logArea.setFont(new Font("Consolas",Font.PLAIN,11));
        bp.add(new JScrollPane(logArea));
        main.add(bp, BorderLayout.SOUTH);

        setContentPane(main);
    }

    private JLabel label(String t, Color c, int sz, boolean bold) {
        JLabel l = new JLabel(t);
        l.setFont(new Font("Segoe UI", bold?Font.BOLD:Font.PLAIN, sz));
        l.setForeground(c);
        l.setAlignmentX(LEFT_ALIGNMENT);
        return l;
    }
    private JLabel statLabel(String t, Color c) {
        JLabel l = new JLabel(t);
        l.setFont(new Font("Segoe UI",Font.BOLD,15));
        l.setForeground(c);
        l.setAlignmentX(LEFT_ALIGNMENT);
        return l;
    }
    private JButton btn(String t, Color bg, ActionListener al) {
        JButton b = new JButton(t);
        b.setFont(new Font("Segoe UI",Font.BOLD,13));
        b.setForeground(Color.WHITE);
        b.setBackground(bg);
        b.setFocusPainted(false);
        b.setBorderPainted(false);
        b.setMaximumSize(new Dimension(240,33));
        b.setAlignmentX(LEFT_ALIGNMENT);
        b.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        b.addActionListener(al);
        return b;
    }
    private void log(String m) {
        SwingUtilities.invokeLater(()->{
            logArea.append("["+java.time.LocalTime.now().withNano(0)+"] "+m+"\n");
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
    }

    private void startVideoThread() {
        Thread t = new Thread(()->{
            log("Connecting to video...");
            while(running) {
                try {
                    URL url = new URL(BACKEND+"/video_feed");
                    HttpURLConnection c = (HttpURLConnection) url.openConnection();
                    c.setConnectTimeout(5000);
                    InputStream is = c.getInputStream();
                    ByteArrayOutputStream buf = new ByteArrayOutputStream();
                    boolean inImg = false;
                    int b, prev=0;
                    while(running && (b=is.read())!=-1) {
                        if(!inImg) {
                            if(prev==0xFF && b==0xD8) {
                                inImg=true; buf.reset();
                                buf.write(0xFF); buf.write(0xD8);
                            }
                        } else {
                            buf.write(b);
                            if(prev==0xFF && b==0xD9) {
                                inImg=false;
                                BufferedImage img = ImageIO.read(new ByteArrayInputStream(buf.toByteArray()));
                                if(img!=null) {
                                    int lw=videoLabel.getWidth(), lh=videoLabel.getHeight();
                                    if(lw>0 && lh>0) {
                                        Image sc = img.getScaledInstance(lw,lh,Image.SCALE_FAST);
                                        SwingUtilities.invokeLater(()->videoLabel.setIcon(new ImageIcon(sc)));
                                    }
                                }
                            }
                        }
                        prev=b;
                    }
                    is.close();
                } catch(Exception e) {
                    SwingUtilities.invokeLater(()->videoLabel.setText("Reconnecting..."));
                    try{Thread.sleep(3000);}catch(InterruptedException ex){}
                }
            }
        },"Video");
        t.setDaemon(true); t.start();
    }

    private void startStatsThread() {
        Thread t = new Thread(()->{
            while(running) {
                try {
                    URL url = new URL(BACKEND+"/stats");
                    HttpURLConnection c = (HttpURLConnection) url.openConnection();
                    BufferedReader r = new BufferedReader(new InputStreamReader(c.getInputStream()));
                    StringBuilder sb = new StringBuilder();
                    String ln; while((ln=r.readLine())!=null) sb.append(ln);
                    r.close();
                    JSONObject j = new JSONObject(sb.toString());
                    SwingUtilities.invokeLater(()->{
                        totalLabel.setText("Total: "+j.getInt("total"));
                        carLabel.setText("Cars: "+j.getInt("car"));
                        bikeLabel.setText("Bikes: "+j.getInt("bike"));
                        busLabel.setText("Buses: "+j.getInt("bus"));
                        truckLabel.setText("Trucks: "+j.getInt("truck"));
                        modeLabel.setText("Mode: "+j.optString("mode","SVM")+"  ");
                        elapsedLabel.setText((int)j.optDouble("elapsed_seconds",0)+"s  ");
                        String d = j.optString("density","Low");
                        densityLabel.setText(d.toUpperCase());
                        switch(d) {
                            case "High": densityPanel.setBackground(new Color(220,30,30)); break;
                            case "Medium": densityPanel.setBackground(new Color(220,180,0)); break;
                            default: densityPanel.setBackground(new Color(0,180,0));
                        }
                    });
                    Thread.sleep(1000);
                } catch(Exception e) { try{Thread.sleep(3000);}catch(InterruptedException ex){} }
            }
        },"Stats");
        t.setDaemon(true); t.start();
    }

    private void setMode(String mode) {
        new Thread(()->{
            try {
                URL u = new URL(BACKEND+"/set_mode");
                HttpURLConnection c = (HttpURLConnection) u.openConnection();
                c.setRequestMethod("POST");
                c.setRequestProperty("Content-Type","application/json");
                c.setDoOutput(true);
                c.getOutputStream().write(("{\"mode\":\""+mode+"\"}").getBytes());
                c.getResponseCode();
                log("Switched to "+mode.toUpperCase());
            } catch(Exception e) { log("Mode error: "+e.getMessage()); }
        }).start();
    }
    private void reset() {
        new Thread(()->{
            try {
                URL u = new URL(BACKEND+"/reset");
                HttpURLConnection c = (HttpURLConnection) u.openConnection();
                c.setRequestMethod("POST"); c.setDoOutput(true);
                c.getOutputStream().write("{}".getBytes());
                c.getResponseCode();
                log("Counters reset.");
            } catch(Exception e) { log("Reset error: "+e.getMessage()); }
        }).start();
    }
    private void upload() {
        JFileChooser fc = new JFileChooser();
        fc.setFileFilter(new javax.swing.filechooser.FileNameExtensionFilter("Video","mp4","avi","mkv"));
        if(fc.showOpenDialog(this)==JFileChooser.APPROVE_OPTION) {
            File f = fc.getSelectedFile();
            log("Uploading "+f.getName()+"...");
            new Thread(()->{
                try {
                    String bnd = "---Bnd"+System.currentTimeMillis();
                    URL u = new URL(BACKEND+"/set_video");
                    HttpURLConnection c = (HttpURLConnection) u.openConnection();
                    c.setRequestMethod("POST");
                    c.setRequestProperty("Content-Type","multipart/form-data; boundary="+bnd);
                    c.setDoOutput(true);
                    OutputStream os = c.getOutputStream();
                    os.write(("--"+bnd+"\r\nContent-Disposition: form-data; name=\"video\"; filename=\""+f.getName()+"\"\r\nContent-Type: video/mp4\r\n\r\n").getBytes());
                    FileInputStream fis = new FileInputStream(f);
                    byte[] buf = new byte[8192]; int len;
                    while((len=fis.read(buf))!=-1) os.write(buf,0,len);
                    fis.close();
                    os.write(("\r\n--"+bnd+"--\r\n").getBytes());
                    os.close();
                    log(c.getResponseCode()==200?"Upload OK":"Upload failed");
                } catch(Exception e) { log("Upload error: "+e.getMessage()); }
            }).start();
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(()->new TrafficGUI().setVisible(true));
    }
}
