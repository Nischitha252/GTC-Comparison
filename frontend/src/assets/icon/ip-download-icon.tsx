// abbot-download-icon.tsx
import React from 'react';
import {Size, getSizeDimensions} from './iconUtils';

interface DownloadIconProps {
  className?: string; // Optional className prop
  size?: Size; // Optional size prop
}

const DownloadIcon: React.FC<DownloadIconProps> = ({  className, size = "medium" }) => {
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
            {`.download-cls-1 { fill: none; } .download-cls-2 { fill: currentColor; }`} {/* Define your styles here */}
            </style>
        </defs>

        <title>ABBoT Download</title>

        <g id="Box">
            <rect className="download-cls-1" width="32" height="32"/>
        </g>
        <g id="Final_icons_-_Common" data-name="Final icons - Common">
            <polygon className="download-cls-2" points="24 15 22 13 18 17 18 3 14 3 14 17 10 13 8 15 16 23 24 15"/>
            <path className="download-cls-2" d="M27,20v5H5V20H2v6a2,2,0,0,0,2,2H28a2,2,0,0,0,2-2V20Z"/>
        </g>
    </svg>
    );
};

export default DownloadIcon;
