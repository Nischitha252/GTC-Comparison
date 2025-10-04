// ip-leftArrow-icon.tsx
import React from 'react';
import {Size, getSizeDimensions} from './iconUtils';

interface LeftArrowIconProps {
    className?: string; // Optional className prop
    size?: Size; // Optional size prop
}

const LeftArrowIcon: React.FC<LeftArrowIconProps> = ({  className, size = "medium" }) => {
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
                    {`.leftArrow-cls-1 { fill: none; } .leftArrow-cls-2 { fill: currentColor; }`} {/* Define your styles here */}
                </style>
            </defs>

            <title>IP Left Arrow</title>
            
            <g id="Box">
                <rect className="leftArrow-cls-1" width="32" height="32" />
            </g>

            <g id="Final_icons_-_Common" data-name="Final icons - Common">
                <polygon className="leftArrow-cls-2" points="29 14 8.55 14 15 7.55 13 5.55 3 15.55 13 25.55 15 23.55 8.45 17 29 17 29 14" />
            </g>
        </svg>
    );
};

export default LeftArrowIcon;
