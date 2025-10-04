// abbot-logout-icon.tsx
import React from 'react';
import {Size, getSizeDimensions} from './iconUtils';

interface LogOutIconProps {
  className?: string; // Optional className prop
  size?: Size; // Optional size prop
}

const LogOutIcon: React.FC<LogOutIconProps> = ({  className, size = "medium" }) => {
  const { width, height } = getSizeDimensions(size);

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 32 32"
      width={width}
      height={height}
      className={className}
    >
        <defs>
            <style>
                {`.logout-cls-1 { fill: none; } .logout-cls-2 { fill: currentColor; stroke: currentColor; stroke-width: 1; }`} {/* Define your styles here */}
            </style>
        </defs>

        <title>ABBoT Logout</title>

        <g id="Box">
          <rect className="logout-cls-1" width="32" height="32"/>
        </g>
        <g id="Final_icons_-_Common" data-name="Final icons - Common">
          <path className="logout-cls-2" d="M6,2H26a1,1,0,0,1,1,1V8H24V5H8V27H24V24h3v5a1,1,0,0,1-1,1H6a1,1,0,0,1-1-1V3A1,1,0,0,1,6,2Z"/>
          <polygon className="logout-cls-2" points="20.56 7 29.12 15.56 20.56 24.12 18.44 22 23.44 17 10 17 10 14 23.32 14 18.44 9.12 20.56 7"/>
        </g>
    </svg>
    );
};

export default LogOutIcon;
