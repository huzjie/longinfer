{{- define "longinfer.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "longinfer.labels" -}}
app: {{ include "longinfer.fullname" . }}
{{- end }}
