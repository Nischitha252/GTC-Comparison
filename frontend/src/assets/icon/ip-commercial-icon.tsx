// ip-commercial-icon.tsx
import React from 'react';
import {Size, getSizeDimensions} from './iconUtils';

interface CommercialIconProps {
    className?: string; // Optional className prop
    size?: Size; // Optional size prop
}

const CommercialIcon: React.FC<CommercialIconProps> = ({  className, size = "medium" }) => {
  const { width, height } = getSizeDimensions(size);

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 28 28"
      width={width}
      height={height}
      className={className}
    >
        <defs>
            <style>
            {`.commercial-cls-1 { fill: none; } .commercial-cls-2 { fill: currentColor; }`} {/* Define your styles here */}
            </style>
        </defs>

        <title>IP Commercial</title>

        <g id="Box">
            <rect className="commercial-cls-1" width="32" height="32"/>
        </g>
        <g id="Final_icons_-_Common" data-name="Final icons - Common">
            <path className="commercial-cls-2" d="M16,3,4,8V23l12,6,12-6V8Zm7.09,6.2-6.93,3.15L9.08,9.13,16,6.25ZM7,11.48l7.5,3.41v10L7,21.15ZM17.5,24.9V15L25,11.63v9.52Z"/>
        </g>
    </svg>
  );
};

export default CommercialIcon;
