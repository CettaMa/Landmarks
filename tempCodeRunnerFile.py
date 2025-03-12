if self.alert_start_time and (current_time - self.alert_start_time <= 5):
                    text_color = (0, 0, 255)  # Red
                    threading.Thread(target=playsound, args=("assets/alerts.mp3",), daemon=True).start()