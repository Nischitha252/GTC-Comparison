// abbot-error-icon.tsx
import React from 'react';
import {Size, getSizeDimensions} from './iconUtils';

interface ErrorIconProps {
  className?: string; // Optional className prop
  size?: Size; // Optional size prop
}

const ErrorIcon: React.FC<ErrorIconProps> = ({  className, size = "medium" }) => {
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
            {`.error-cls-1 { fill: none; } .error-cls-2 { fill: currentColor; }`} {/* Define your styles here */}
            </style>
        </defs>

        <title>ABBoT Error</title>

        <g id="Box">
            <rect className="error-cls-1" width="32" height="32"/>
        </g>
        <g id="Final_icons_-_Common" data-name="Final icons - Common">
            <path className="error-cls-2" d="M16,2A14,14,0,1,0,30,16,14,14,0,0,0,16,2Zm6.06,17.94-2.12,2.12L16,18.12l-3.94,3.94L9.94,19.94,13.88,16,9.94,12.06l2.12-2.12L16,13.88l3.94-3.94,2.12,2.12L18.12,16Z"/>
        </g>
    </svg>
    );
};

export default ErrorIcon;
