import { useDarkStore } from "@/stores/darkStore";
import React, { forwardRef } from "react";
import SvgPineconeLogo from "./PineconeLogo";

export const PineconeIcon = forwardRef<
  SVGSVGElement,
  React.PropsWithChildren<{}>
>((props, ref) => {
  const isDark = useDarkStore((state) => state.dark);

  return (
    <SvgPineconeLogo ref={ref} {...props} color={isDark ? "#fff" : "#000"} />
  );
});
