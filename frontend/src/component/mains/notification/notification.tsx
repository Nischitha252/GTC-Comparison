import React from "react";
import "./notification.css";

// Define the props type
interface NotificationProps {
  message: string;
  color: string;
}

const Notification: React.FC<NotificationProps> = ({ message, color }) => {
  return (
    <div className="notification">
      <div
        className="notification-header"
        style={{ backgroundColor: color }}
      ></div>
      <div className="notification-content">
        <p className="message">{message}</p>
      </div>
    </div>
  );
};

export default Notification;
