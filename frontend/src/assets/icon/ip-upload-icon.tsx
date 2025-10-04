// ip-upload-icon.tsx
import React from 'react';
import {Size, getSizeDimensions} from './iconUtils';

interface UploadIconProps {
    className?: string; // Optional className prop
    size?: Size; // Optional size prop
}

const UploadIcon: React.FC<UploadIconProps> = ({  className, size = "medium" }) => {
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
            {`.upload-cls-1 { fill: none; } .upload-cls-2 { fill: currentColor; }`} {/* Define your styles here */}
            </style>
        </defs>

        <title>IP Upload</title>

        <g id="Box">
            <rect className="upload-cls-1" width="32" height="32" />
        </g>
        
        <g id="Final_icons_-_Common" data-name="Final icons - Common">
            <polygon className="upload-cls-2" points="14 9 14 23 18 23 18 9 22 13 24 11 16 3 8 11 10 13 14 9" />
            <path className="upload-cls-2" d="M27,20v5H5V20H2v6a2,2,0,0,0,2,2H28a2,2,0,0,0,2-2V20Z" />
        </g>
    </svg>
  );
};

export default UploadIcon;
